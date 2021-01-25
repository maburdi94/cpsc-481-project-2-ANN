
import utils
import re
import sys
import math
import numpy as np


# Sigmoid (activation function)
def sigmoid (x) : 
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x) *(1-sigmoid (x))



# Read classified set and target values from input file
def read_input(filename = "input.txt"):
    file 	 = open(filename, "r")

    # Load the pre-classified vectors from file and the targets
    pre 	 = np.empty((0,10))
    targets  = np.empty((0,1))

    for line in file:
        group     = re.search("\( *(\d+) *\(((?:\d+ *)+) *\) *(\d+)\)", line).groups()

        _id  	  = int(group[0])
        _features = [int(i) for i in group[1].split(' ')]
        _class 	  = int(group[2])

        pre = np.append(pre, [_features], axis=0)
        targets = np.append(targets, [[_class]], axis=0)

    file.close()

    return [pre, targets]








# Print entire array
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=1000)


HIDDEN_LAYER_NODES = 12

# Read vectors and target classes from file
[pre, expected] = read_input()


# 80% of inputs is training set
training_set = pre[0:math.floor(.8*len(pre)),:]
# print("Training Set\n", training_set)

# 20% is holdout set
holdout_set = pre[math.floor(.8*len(pre)):,:]
# print("Holdout Set\n", holdout_set)


targets = np.full((len(training_set), 8), 0.2)

for i,(t,[e]) in enumerate(zip(targets, expected)):
    targets[i,int(e)] = 0.8


epoch = 1



wh = np.array([[-2.07496542e-01, -5.75604499e-01, -5.81001895e-01,  1.18496005e+00, -3.94035298e-02,  5.98370979e-01,  1.37435302e+00,  1.50198791e-01,  1.56342191e+00, -9.37469799e-01, -2.30456349e-01, -3.63012781e-01],
[-8.71368337e-01,  2.95085062e-01, -1.07550873e+00,  2.37718571e-01, -8.42622597e-01,  1.67375713e+00, -2.57045524e+00,  2.64353633e-01, -1.32245018e+00,  6.31207511e-01,  1.39906473e+00, -4.61418167e-01],
[-5.89594874e-01,  1.83652836e-01, -3.12261934e-01, -6.73285364e-01,  2.23548225e-01,  6.38290045e-02,  2.12048816e-01,  1.25506736e-01, -9.27617095e-01, -1.26184480e+00, -2.96801088e-01,  3.01497724e-01],
[ 1.51802933e+00, -1.30687132e+00, -1.16659643e+00,  1.13031076e-01,  8.39420494e-01, -1.69262287e+00,  7.66133584e-01, -3.39649386e-01,  1.51119111e-01,  1.45013033e+00, -1.14738212e+00, -2.66626318e-01],
[ 2.54337559e-04, -2.22171053e-01, -1.06528952e-02,  2.33794294e-02,  6.62020850e-01, -1.99012188e-01, -9.73972954e-04, -3.01400549e-03,  1.47329969e-02,  3.33357855e-01,  3.19459393e-04, -4.04416664e-01],
[ 1.19802950e+00,  6.41881988e-01,  1.50842296e+00,  1.86771148e-01,  7.86358531e-01,  3.29395282e-01,  1.46710099e+00,  2.83264619e-01,  3.77097139e-01,  7.41300292e-01, -7.97083170e-02,  3.46232265e-01],
[ 3.23761602e-01,  9.80431616e-01, -6.32254182e-01, -1.40086309e+00, -4.40015435e-01, -4.31377022e-01,  1.24200415e+00, -1.88824806e-01, -6.42792207e-01, -6.83182056e-01, -1.09621300e+00, -3.10251861e-01],
[-9.23596246e-02,  1.32992312e+00,  8.29304603e-02,  9.07601458e-01,  1.05563074e+00,  7.10046181e-01,  2.41936312e-01, -1.29016978e-01, -1.04619571e+00, -1.47100812e+00,  8.28651605e-01,  1.12593368e-01],
[ 1.28958513e+00,  1.44022136e+00,  2.61933371e-01,  5.74267022e-01, -2.88661396e-01, -1.16286578e+00, -1.12594952e+00, -5.80674906e-02, -1.79929131e-01,  2.23046353e+00, -1.39384544e+00,  8.44634333e-02],
[-1.12727512e+00, -1.12158522e+00,  4.19701253e-01,  1.57466281e+00, -4.36301311e-02,  2.23719439e+00, -3.59670138e-01,  1.13223031e+00,  8.03604961e-02,  1.26705782e+00,  3.20735532e-01,  2.19909142e-01]])

wo = np.array([[-0.53995259, -0.75596075, -1.47395686,  1.67004859,  2.16536408, -1.73902644,  0.44591619, -0.26507096],
[ 0.46884148,  0.64242429,  0.17980081, -0.59826671, -1.65965066,  0.96304058,  0.41432411,  0.06306921],
[-0.388879  , -0.6690064 , -1.19738381, -0.64631892,  2.31246541, -0.06349733,  0.54632434, -0.27059701],
[ 0.44997387,  1.23239005,  0.85664566, -0.97914678, -0.88821986,  0.52533043, -0.59053746, -0.05616684],
[ 0.09103122, -0.77708226,  1.60077363,  0.89280294, -2.06635727,  1.63475923, -1.33162943,  0.33980205],
[ 0.40225291, -1.42621443,  1.07022404,  0.68351257, -1.50593787,  0.50002733,  0.82160227, -0.12228729],
[ 0.21188103,  0.03260993, -0.88211112,  0.13517171,  1.14558354,  0.32356191, -1.08265912,  0.1703306 ],
[ 0.18611894,  2.25427998, -0.04425822, -0.42574258,  0.99851948, -2.8311664 , -0.20484569,  0.04758283],
[-0.26732475, -0.61969337, -0.02671346,  2.93551432, -1.89750935, -0.43412129,  0.03139263,  0.02462607],
[ 0.65498153,  0.61956551,  1.03842399, -2.37989918, -0.17445406,  0.61989005,  0.18127994, -0.15438645],
[-0.5751508 , -0.67548377, -0.29829939, -0.52188074,  2.8914635 , -0.95868671, -0.5269937 ,  0.25654102],
[ 0.89265166,  0.70970829,  0.23881408,  0.69057178, -4.46682803,  0.57846611, -0.85097825,  2.56132703]])

bh = np.array(  [ 0.62894607,  0.28282301, -0.9197654 , -0.47688999, -1.17370858, -1.63258731, -0.81710027,  1.34419765,  3.57397343, -0.46493966, -0.91439044, -2.10770023]    )

bo = np.array(  [-2.67474091, -2.62329891, -3.02180466, -0.05222304,  0.48894164, -0.60440281,  0.18265409, -1.45387116] ) 


# wh = np.random.uniform(-.3, .3, (10, HIDDEN_LAYER_NODES))
# wo = np.random.uniform(-.3, .3, (HIDDEN_LAYER_NODES, 8))
# bh = np.random.randn(HIDDEN_LAYER_NODES)
# bo = np.random.randn(8)




def main():
    global wh
    global wo
    global bo
    global bh
    global epoch

    cost = math.inf

    error = 1000.0

    desired_error = 0.1

    LR = 8e-4

    error_cost = []

    # Training
    while epoch < 10000000:

        zh = np.dot(training_set, wh) + bh
        ah = sigmoid( zh )                                   # [SET_SIZE x HIDDEN_LAYER_NODES]
        
        zo = np.dot(ah, wo) + bo
        ao = sigmoid( zo )                                   # [SET_SIZE x 8]

    
        dcost_dzo = targets - ao                             # [SET_SIZE x 8]
        dzo_dwo   = ah
        dcost_wo = np.dot(dzo_dwo.T, dcost_dzo) 

        dcost_bo = dcost_dzo


        dzo_dah = wo
        dcost_dah = np.dot(dcost_dzo , dzo_dah.T)
        dah_dzh = sigmoid_der(zh)
        dzh_dwh = training_set
        dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)

        dcost_bh = dcost_dah * dah_dzh


        wh += LR * dcost_wh
        bh += LR * dcost_bh.sum(axis=0)

        wo += LR * dcost_wo
        bo += LR * dcost_bo.sum(axis=0)



        if epoch % 500 == 0 and cost > 1e-5:
            cost = np.sum((targets - ao)**2)/len(targets)
            print('Cost (MSE): ', cost, "\tLR: ", LR)
            error_cost.append(cost)

            if cost < 0.1: break

        # Adaptive learning rate decrese by 0.01 every 100 epochs
        # if (epoch % 100000 == 0):
        #     LR = LR * math.exp(-0.04*(epoch/100000))



        epoch += 1


    print("Epochs: ", epoch)
    print("Layer 1\nBias: \n", np.array2string(bh, separator=', '), "\nWeights: \n,", np.array2string(wh, separator=', '))
    print()
    print("Layer 2\nBias: \n", np.array2string(bo, separator=', '), "\nWeights: \n,", np.array2string(wo, separator=', '))





if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Stopped')
        print("Epochs: ", epoch)
        print("Layer 1\nBias: \n", np.array2string(bh, separator=', '), "\nWeights: \n,", np.array2string(wh, separator=', '))
        print()
        print("Layer 2\nBias: \n", np.array2string(bo, separator=', '), "\nWeights: \n,", np.array2string(wo, separator=', '))

        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
