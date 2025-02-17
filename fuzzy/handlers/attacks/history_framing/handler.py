import json
import logging

from fuzzy.handlers.attacks.base import attack_handler_fm
from fuzzy.handlers.attacks.enums import FuzzerAttackMode
from fuzzy.handlers.attacks.taxonomy.handler import TaxonomyParaphraser

logger = logging.getLogger(__name__)

@attack_handler_fm.flavor(FuzzerAttackMode.HISTORY_FRAMING)
class HistoryFramingAttackhandler(TaxonomyParaphraser):
    """
    Taxonomy Paraphraser attack technique (https://arxiv.org/pdf/2401.06373)
    """

    def _load_taxonomy_dataset(self) -> None:
        with open('resources/persuasion_taxonomy.jsonl', 'r') as f:
            data = f.read()

        self._taxonomies = [json.loads(jline) for jline in data.splitlines()][:1]
    

        