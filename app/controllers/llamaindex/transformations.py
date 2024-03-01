import re
from llama_index.core.schema import TransformComponent


class TextCleaner(TransformComponent):
    '''
    To remove \n found in the text.
    '''
    def __call__(self, nodes, **kwargs):
        for node in nodes:
            node.text = re.sub(r"\n", ". ", node.text)
        return nodes
    
