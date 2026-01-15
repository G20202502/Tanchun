class ConceptManager:
    def __init__(self, concept_keys):
        """
        concept_keys: List of concept keys, in fixed order
        e.g., "is high request rate", "is continuous requesting", "is fixed behavior pattern"]
        """
        self.concept_keys = concept_keys
        self.num_bits = len(concept_keys)
        self.max_id = 2 ** self.num_bits

    def dict_to_id(self, concept_dict):
        """将概念字典编码为唯一的 concept_id"""
        bits = [int(concept_dict[key]) for key in self.concept_keys]
        concept_id = 0
        for i, bit in enumerate(bits[::-1]):  # reverse to match bit order
            concept_id += bit << i
        return concept_id

    def id_to_dict(self, concept_id):
        """将 concept_id 解码为概念字典"""
        if not (0 <= concept_id < self.max_id):
            raise ValueError(f"concept_id 必须在 0 到 {self.max_id - 1} 之间")

        bits = [(concept_id >> i) & 1 for i in reversed(range(self.num_bits))]
        return {
            key: bool(bit) for key, bit in zip(self.concept_keys, bits)
        }

    def num_concepts(self):
        return self.max_id
    

