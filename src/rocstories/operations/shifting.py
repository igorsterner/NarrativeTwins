from src.rocstories.operations import base


class Shifting(base.Operation):

    @staticmethod
    def variants(sentences, s):

        n = len(sentences)
        item = sentences[s]
        base_seq = sentences[:s] + sentences[s + 1 :]

        variants = []
        for t in range(n):
            if t == s:
                continue
            new_seq = base_seq[:t] + [item] + base_seq[t:]

            variants.append(" ".join(new_seq))

        return variants

    def scores(self, sentences):

        base_text = " ".join(sentences)
        base_emb = self.encoder.encode([base_text])[0]

        scores = []
        n = len(sentences)
        for s in range(n):

            texts = self.variants(sentences, s)
            cand_embs = self.encoder.encode(texts)

            sims = (cand_embs @ base_emb) / (cand_embs.norm(dim=1) * base_emb.norm())

            m = sims.mean().item()

            scores.append(1.0 - m)

        return scores
