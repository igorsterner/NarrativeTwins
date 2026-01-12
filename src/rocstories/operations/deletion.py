from src.rocstories.operations import base


class Deletion(base.Operation):

    @staticmethod
    def variants(sentences):

        variants = []
        for i in range(len(sentences)):
            v = sentences[:i] + sentences[i + 1 :]
            variants.append(" ".join(v))

        return variants

    def scores(self, sentences):

        base_text = " ".join(sentences)

        variants = self.variants(sentences)

        texts = [base_text] + variants
        emb = self.encoder.encode(texts)

        sims = (emb[1:] @ emb[0]) / (emb[1:].norm(dim=1) * emb[0].norm())

        sal = (1.0 - sims).tolist()

        return sal
