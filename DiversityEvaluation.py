import numpy as np

class DiversityEvaluator:
    def __init__(self):
        self.dist1 = []
        self.dist2 = []
        self.dist3 = []

    def compute(self, generations_batch):
        self.dist1.clear()
        self.dist2.clear()
        self.dist3.clear()

        for generations in generations_batch:
            unigrams, bigrams, trigrams = set(), set(), set()
            total_words = 0

            for gen in generations:
                tokens = gen.split()
                total_words += len(tokens)
                unigrams.update(tokens)

                for i in range(len(tokens) - 1):
                    bigrams.add(f"{tokens[i]}_{tokens[i + 1]}")

                for i in range(len(tokens) - 2):
                    trigrams.add(f"{tokens[i]}_{tokens[i + 1]}_{tokens[i + 2]}")

            if total_words > 0:
                self.dist1.append(len(unigrams) / total_words)
                self.dist2.append(len(bigrams) / total_words)
                self.dist3.append(len(trigrams) / total_words)
            else:
                # 防止除以零
                self.dist1.append(0.0)
                self.dist2.append(0.0)
                self.dist3.append(0.0)

        return (
            np.nanmean(self.dist1),
            np.nanmean(self.dist2),
            np.nanmean(self.dist3),
        )

# # 使用示例
# if __name__ == "__main__":
#     # 假设每个 prompt 有多个生成
#     sample_data = [
#         ["the cat sat on the mat", "a cat sat on a mat"],
#         ["this is great", "this is awesome", "this is fantastic"],
#     ]
#
#     metrics = DiversityMetrics()
#     dist1, dist2, dist3 = metrics.compute(sample_data)
#     print(f"Dist-1: {dist1:.4f}, Dist-2: {dist2:.4f}, Dist-3: {dist3:.4f}")
