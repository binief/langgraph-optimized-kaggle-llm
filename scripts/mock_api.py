from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Demo Business API")

CUSTOMERS = {
    "C-101": {"id": "C-101", "name": "Asha Menon", "tier": "enterprise", "status": "active"},
    "C-102": {"id": "C-102", "name": "Ravi Kumar", "tier": "standard", "status": "suspended"},
}

ORDERS = {
    "ORD-500": {"order_id": "ORD-500", "customer_id": "C-101", "amount": 1299.00, "status": "delayed"},
    "ORD-501": {"order_id": "ORD-501", "customer_id": "C-102", "amount": 89.00, "status": "shipped"},
}

TICKETS = []

class TicketIn(BaseModel):
    customer_id: str
    priority: str
    summary: str

@app.get("/customers/{customer_id}")
def get_customer(customer_id: str):
    if customer_id not in CUSTOMERS:
        raise HTTPException(404, f"Customer {customer_id} not found")
    return CUSTOMERS[customer_id]

@app.get("/orders/{order_id}")
def get_order(order_id: str):
    if order_id not in ORDERS:
        raise HTTPException(404, f"Order {order_id} not found")
    return ORDERS[order_id]

@app.post("/tickets")
def create_ticket(t: TicketIn):
    if t.priority not in ("low", "medium", "high"):
        raise HTTPException(400, "priority must be low, medium, or high")
    ticket = {"ticket_id": f"TKT-{1000 + len(TICKETS)}", **t.model_dump()}
    TICKETS.append(ticket)
    return ticket