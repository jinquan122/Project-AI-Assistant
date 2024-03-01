from llama_index.core.response_synthesizers import TreeSummarize, Refine

class ResponseSynthesizer:
    def __init__(self, verbose=False):
        self.verbose = verbose
        
    def tree_summarize(self, llm):
        return TreeSummarize(verbose=self.verbose, llm=llm)
    
    def refine_summarize(self):
        return Refine(verbose=self.verbose)
