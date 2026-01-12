from src.rocstories.operations import base


class Disruption(base.Operation):

    @staticmethod
    def variants(sentences):
        variants = []
        n = len(sentences)
        for k in range(n + 1):
            prefix = sentences[:k]
            variant = " ".join(prefix)
            variants.append(variant)

        return variants

    def scores(self, sentences):

        texts = self.variants(sentences)

        emb = self.encoder.encode(texts)

        scores = []
        for i in range(len(sentences)):
            if i == 0:
                scores.append(0.0)
                continue

            sim = (emb[i + 1] @ emb[i]) / (emb[i + 1].norm() * emb[i].norm())

            scores.append(1.0 - sim.item())

        return scores
