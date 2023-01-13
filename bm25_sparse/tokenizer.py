class BaseTokenizer:
    """
    Base Tokenizer
    """
    def preprocessing(self, text:str) -> str:
        raise NotImplementedError()
    
    def _tokenize(self, text:str) -> list:
        raise NotImplementedError()

    def tokenize(self, text:str) -> list:
        text = self.preprocessing(text)
        return self._tokenize(text)

class Tokenizer(BaseTokenizer):
    """
    White Space Tokenizer
    """
    def preprocessing(self, text:str):
        text = text.lower()
        return text
    def _tokenize(self, text: str):
        return text.split()