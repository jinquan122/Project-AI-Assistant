# setup Arize Phoenix for logging/observability
import phoenix as px
import llama_index.core

class LLM_observability:
    def __init__(self):
        self.phoenix = "arize_phoenix"
        
    def phoenix_observability(self):
        '''
        To perform LLM observability.
        '''
        px.launch_app()
        llama_index.core.set_global_handler(self.phoenix)
        return px