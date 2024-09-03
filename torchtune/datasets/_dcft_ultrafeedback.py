from typing import Any, Dict, Mapping, Optional

from torchtune.data import Message
from torchtune.datasets._preference import PreferenceDataset
from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.modules.transforms import Transform

class UltraFeedbackToMessages(Transform):
    def __init__(self, train_on_input: bool = False):
        self.train_on_input = train_on_input

    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        chosen_messages = [
            Message(
                role="user", content=sample["chosen"][0]["content"], masked=not self.train_on_input
            ),
            Message(role="assistant", content=sample["chosen"][1]["content"]),
        ]
        rejected_messages = [
            Message(
                role="user", content=sample["rejected"][0]["content"], masked=not self.train_on_input
            ),
            Message(role="assistant", content=sample["rejected"][1]["content"]),
        ]
        return {"chosen": chosen_messages, "rejected": rejected_messages}
    
def ultrafeedback_dataset(
    tokenizer: ModelTokenizer,
    *,
    source: str = "argilla/ultrafeedback-binarized-preferences-cleaned",
    train_on_input: bool = False,
    split: str = "train",
) -> PreferenceDataset:
    message_transform = UltraFeedbackToMessages(train_on_input=train_on_input)
    return PreferenceDataset(
        source=source,
        message_transform=message_transform,
        tokenizer=tokenizer,
        split=split,
    )