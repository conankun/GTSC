import util

para = util.algPara(0.8, 1, 5, 0.4)

P = util.read_tensor("data/test.tns")

(r, h) = util.tensor_speclustering(P, para)