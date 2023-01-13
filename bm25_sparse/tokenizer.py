class BaseTokenizer:
    """
    Base Tokenizer
    """
    def preprocessing(self, text:str) -> str:
        raise NotImplementedError()
    
    def _tokenizer(self, text:str) -> list:
        raise NotImplementedError()

    def tokenizer(self, text:str) -> list:
        text = self.preprocessing(text)
        return self._tokenizer(text)

class Tokenizer(BaseTokenizer):
    """
    White Space Tokenizer
    """
    def preprocessing(self, text:str):
        text = text.lower()
        return text
    def _tokenizer(self, text: str):
        return text.split()