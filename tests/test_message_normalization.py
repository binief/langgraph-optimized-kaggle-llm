import unittest

from scripts.message_normalization import merge_system_messages


class MergeSystemMessagesTests(unittest.TestCase):
    def test_moves_and_merges_system_messages_without_reordering_conversation(self):
        messages = [
            {"role": "user", "content": "question"},
            {"role": "system", "content": "late instruction"},
            {"role": "assistant", "content": "prior reply"},
            {"role": "system", "content": "another instruction"},
            {"role": "tool", "content": "tool result"},
        ]

        result = merge_system_messages(messages)

        self.assertEqual([message["role"] for message in result], ["system", "user", "assistant", "tool"])
        self.assertEqual(
            result[0]["content"],
            "late instruction\n\nanother instruction",
        )
        self.assertEqual(result[1:], [messages[0], messages[2], messages[4]])

    def test_preserves_non_text_system_content_as_json(self):
        result = merge_system_messages([
            {"role": "user", "content": "question"},
            {"role": "system", "content": [{"type": "text", "text": "rule"}]},
        ])

        self.assertEqual(result[0]["role"], "system")
        self.assertEqual(result[0]["content"], '[{"type": "text", "text": "rule"}]')

    def test_does_not_create_a_system_message_when_absent(self):
        messages = [{"role": "user", "content": "question"}]
        self.assertEqual(merge_system_messages(messages), messages)


if __name__ == "__main__":
    unittest.main()
