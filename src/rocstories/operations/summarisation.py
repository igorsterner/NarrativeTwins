from src.rocstories.operations import base


class Summarisation(base.Operation):

    def scores(self, sentences):

        base_text = " ".join(sentences)
        texts = [base_text] + sentences

        emb = self.encoder.encode(texts)

        sims = (emb[1:] @ emb[0]) / (emb[1:].norm(dim=1) * emb[0].norm())

        return sims.tolist()
