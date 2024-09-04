from typing import Any, Dict, Mapping, Optional

from torchtune.data import Message
from torchtune.datasets._preference import PreferenceDataset
from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.modules.transforms import Transform
from torchtune.datasets._stack_exchange_paired import StackExchangePairedToMessages
    
def ultrafeedback_dataset(
    tokenizer: ModelTokenizer,
    *,
    source: str = "argilla/ultrafeedback-binarized-preferences",
    column_map: Optional[Dict[str, str]] = None,
    train_on_input: bool = False,
    split: str = "train",
) -> PreferenceDataset:
    message_transform = StackExchangePairedToMessages(
        train_on_input=train_on_input, column_map=column_map
    )
     
    return PreferenceDataset(
        source=source,
        message_transform=message_transform,
        tokenizer=tokenizer,
        split=split,
    )