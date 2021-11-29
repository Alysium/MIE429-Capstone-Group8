import matplotlib.pyplot as plt
files = ["rastrigin_500_gp_hedge_seed2.txt","rastrigin_500_gp_hedge_seed1.txt","rastrigin_500_gp_hedge_seed3.txt", \
    "rosenbrock_500_gp_hedge_seed1.txt","rosenbrock_500_gp_hedge_seed2.txt","rosenbrock_500_gp_hedge_seed3.txt", \
    "schwefel_500_gp_hedge_seed1.txt","schwefel_500_gp_hedge_seed2.txt","schwefel_500_gp_hedge_seed3.txt"]
mins = []
iterMins = []
for file in files:
    arr = []
    with open(file) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
        minVal = float('inf')
        for line in lines:
            vals = [float(i) for i in line.split(",")]
            minVal = min(minVal, vals[-1])
            arr.append(minVal)
        print("minVal for", file , ":", minVal)
    mins.append(minVal)
    iterMins.append(arr)

rastriginPlot = iterMins[:3]
rosenbrockPlot = iterMins[3:6]
schwefelPlot = iterMins[6:]

rastriginAvg = sum(mins[:3])/3
rosenbrockAvg = sum(mins[3:6])/3
schwefelAvg = sum(mins[6:])/3

print("rastriginAvg",rastriginAvg)
print("rosenbrockAvg",rosenbrockAvg)
print("schwefelAvg",schwefelAvg)

iters = [i for i in range(500)]

fig = plt.figure()
ax = plt.axes()
plt.title("Bayesian Minimum Value found for Schwefel function")
plt.ylabel("Minimum Value")
plt.xlabel("Iterations")
plt.plot(iters,schwefelPlot[0], color='blue')
plt.plot(iters,schwefelPlot[1], color='orange')
plt.plot(iters,schwefelPlot[2], color='green')
plt.savefig("SchwefelGraph.png")
plt.show()


