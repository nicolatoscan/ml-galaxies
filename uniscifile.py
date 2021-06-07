from collections import Counter
import random


with open("avg.csv", "w") as avgF:
    with open("1.csv") as f1:
        with open("2.csv") as f2:
            with open("3.csv") as f3:
                ll1 = f1.readlines()
                ll2 = f2.readlines()
                ll3 = f3.readlines()

                tot = [0, 0, 0, 0]

                for l1, l2, l3 in zip(ll1, ll2, ll3):
                    c = Counter([l1, l2, l3])
                    keys = c.keys()
                    onlyOne = False
                    done = False

                    tot[0] += 1
                    tot[len(keys)] += 1


                    if len(keys) == 1 or len(keys) == 2:
                        for k in keys:
                            if c[k] == 2 or c[k] == 3:
                                avgF.write(k)
                                # print(k)
                                onlyOne = not onlyOne
                                done = True
                    elif (len(keys) == 3):
                        ks = []
                        for k in keys:
                            ks.append(k)
                        k = random.choice(ks)
                        avgF.write(k)
                        onlyOne = not onlyOne
                        done = True

                    if not onlyOne or not done:
                        print("DIOCAN")


print(tot[1] / tot[0])
print(tot[2] / tot[0])
print(tot[3] / tot[0])



