import copy, numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_otput_to_derivation(output):
    return output*(1-output)

binary_dim=8

largest_number=pow(2,binary_dim)

def int2binary(num):
    a = bin(num)
    a = a[2:len(a)]
    num_bin = np.zeros(binary_dim)

    for i in range(binary_dim - len(a), binary_dim):
        num_bin[i] = int(a[i + len(a) - binary_dim])
    return num_bin


alpha=0.05
input_dim=2
hidden_dim=16
output_dim=1

synapse_0=0.1*np.random.random((input_dim,hidden_dim))-0.05


synapse_1=0.1*np.random.random((hidden_dim,output_dim))-0.05
1
synapse_h=0.1*np.random.random((hidden_dim,hidden_dim))-0.05

synapse_0_update=np.zeros_like(synapse_0)
synapse_1_update=np.zeros_like(synapse_1)
synapse_h_update=np.zeros_like(synapse_h)


for i in range(10000):

    a_int = np.random.randint(largest_number / 2)
    b_int = np.random.randint(largest_number / 2)

    a = int2binary(a_int)
    b = int2binary(b_int)

    c_int = a_int + b_int
    c = int2binary(c_int)

    d = np.zeros_like(c)

    overallError = 0

    layer_2_deltas = list()
    layer_1_values = list()
    layer_1_values.append(np.zeros(hidden_dim))

    for position in range(binary_dim):
        X = np.array([[a[binary_dim - 1 - position], b[binary_dim - 1 - position]]])
        Y = np.array([[c[binary_dim - 1 - position]]]).T

        layer_1 = sigmoid(np.dot(X, synapse_0) + np.dot(layer_1_values[-1], synapse_h))
        layer_2 = sigmoid(np.dot(layer_1, synapse_1))

        layer_2_error = Y - layer_2
        layer_2_deltas.append(layer_2_error * sigmoid_otput_to_derivation(layer_2))


        overallError += np.abs(layer_2_error[0][0])



        d[binary_dim-1-position] = np.round(layer_2[0][0])

        layer_1_values.append(copy.deepcopy(layer_1))

    # what is this used for???
    future_layer_1_delta = np.zeros(hidden_dim)

    for position in range(binary_dim):
        X = np.array([[a[position], b[position]]])
        layer_1 = layer_1_values[-1 - position]
        prev_layer_1 = layer_1_values[-2 - position]

        layer_2_delta = layer_2_deltas[-position - 1]


        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + \
                         layer_2_delta.dot(synapse_1.T)) * sigmoid_otput_to_derivation(layer_1)


        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update += np.atleast_2d(X).T.dot(layer_1_delta)

        future_layer_1_delta=layer_1_delta

    synapse_0+=synapse_0_update
    synapse_1 += synapse_1_update
    synapse_h += synapse_h_update


    synapse_0_update *=0
    synapse_1_update *=0
    synapse_h_update *=0

    if(i%1000==0):
        print "Error: "+str(overallError)
        print "Pred: " +str(d)
        print "true: " +str(c)
        out=0

        for index,x in enumerate(reversed(d)):
            out+=x*pow(2,index)

        print str(a_int)+" + "+str(b_int)+" = "+str(out)



