import copy, numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_otput_to_derivation(output):
    return output*(1-output)


def tanh_derivative(values):
    return 1. - tanh(values) ** 2

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

binary_dim=8

largest_number=pow(2,binary_dim)

def int2binary(num):
    a = bin(num)
    a = a[2:len(a)]
    num_bin = np.zeros(binary_dim)

    for i in range(binary_dim - len(a), binary_dim):
        num_bin[i] = int(a[i + len(a) - binary_dim])
    return num_bin

alpha=0.1
input_dim=2
hidden_dim=26
output_dim=1


synapse_x_i=0.2*np.random.random((input_dim,hidden_dim))-0.1
synapse_U_i=0.2*np.random.random((hidden_dim,hidden_dim))-0.1

synapse_x_f=0.2*np.random.random((input_dim,hidden_dim))-0.1
synapse_U_f=0.2*np.random.random((hidden_dim,hidden_dim))-0.1

synapse_x_o=0.2*np.random.random((input_dim,hidden_dim))-0.1
synapse_U_o=0.2*np.random.random((hidden_dim,hidden_dim))-0.1

synapse_x_new=0.2*np.random.random((input_dim,hidden_dim))-0.1
synapse_U_new=0.2*np.random.random((hidden_dim,hidden_dim))-0.1

synapse_h_o=0.2*np.random.random((hidden_dim,output_dim))-0.1


synapse_x_i_update=np.zeros_like(synapse_x_i)
synapse_U_i_update=np.zeros_like(synapse_U_i)

synapse_x_f_update=np.zeros_like(synapse_x_f)
synapse_U_f_update=np.zeros_like(synapse_U_f)

synapse_x_o_update=np.zeros_like(synapse_x_o)
synapse_U_o_update=np.zeros_like(synapse_U_o)

synapse_x_new_update=np.zeros_like(synapse_x_new)
synapse_U_new_update=np.zeros_like(synapse_U_new)

synapse_h_o_update=np.zeros_like(synapse_h_o)







for epoch in range(20000):
    a_int = np.random.randint(largest_number / 2)
    b_int = np.random.randint(largest_number / 2)

    a = int2binary(a_int)
    b = int2binary(b_int)

    c_int = a_int + b_int
    c = int2binary(c_int)

    d = np.zeros_like(c)

    overallError = 0


    layer_o_deltas = list()
    i_gates=list()
    f_gates=list()
    o_gates=list()
    g_gates=list()  # new memory
    states=list()
    h_outputs = list()

    states.append(np.zeros(hidden_dim))
    h_outputs.append(np.zeros(hidden_dim))

    for position in range(binary_dim):
        X = np.array([[a[binary_dim - 1 - position], b[binary_dim - 1 - position]]])



        Y = np.array([[c[binary_dim - 1 - position]]])

        i_gate=sigmoid(np.dot(X,synapse_x_i)+np.dot(h_outputs[-1],synapse_U_i))
        o_gate=sigmoid(np.dot(X,synapse_x_o)+np.dot(h_outputs[-1],synapse_U_o))
        f_gate=sigmoid(np.dot(X,synapse_x_f)+np.dot(h_outputs[-1],synapse_U_f))
        g_gate=sigmoid(np.dot(X,synapse_x_new)+np.dot(h_outputs[-1],synapse_U_new))
        state=(i_gate*g_gate+states[-1]*f_gate)
        h_output=tanh(state)*o_gate

        output=sigmoid(np.dot(h_output,synapse_h_o))

        layer_o_error = Y - output

        layer_o_deltas.append(layer_o_error * sigmoid_otput_to_derivation(output))


        overallError += np.abs(layer_o_error[0][0])



        d[binary_dim-1-position] = np.round(output[0][0])

        i_gates.append(copy.deepcopy(i_gate))
        o_gates.append(copy.deepcopy(o_gate))
        f_gates.append(copy.deepcopy(f_gate))
        g_gates.append(copy.deepcopy(g_gate))
        states.append(copy.deepcopy(state))
        h_outputs.append(copy.deepcopy(h_output))


    future_o_delta=np.zeros(hidden_dim)
    future_i_delta=np.zeros(hidden_dim)
    future_f_delta=np.zeros(hidden_dim)
    future_g_delta=np.zeros(hidden_dim)
    future_state_delta=np.zeros(hidden_dim)
    future_f_gate=np.zeros(hidden_dim)



    for position in range(binary_dim):
        X = np.array([[a[position], b[position]]])


        i_gate=i_gates[-1-position]
        o_gate=o_gates[-1-position]
        f_gate=f_gates[-1-position]
        g_gate=g_gates[-1-position]





        output_delta=layer_o_deltas[-1-position]

        state = states[-1-position]
        h_output = h_outputs[-1-position]
        h_prev_output=h_outputs[-2-position]
        prev_state=states[-2-position]

        h_delta=(future_o_delta.dot(synapse_U_o.T)+\
                 future_i_delta.dot(synapse_U_i.T)+\
                 future_f_delta.dot(synapse_U_f.T)+\
                 future_g_delta.dot(synapse_U_new.T)+ \
                 output_delta.dot(synapse_h_o.T))

        o_delta=h_delta*tanh(state)*sigmoid_otput_to_derivation(o_gate)

        state_delta=h_delta*o_gate*tanh_derivative(state)+future_state_delta*future_f_gate

        f_delta=state_delta*prev_state*sigmoid_otput_to_derivation(f_gate)

        i_delta=state_delta*g_gate*sigmoid_otput_to_derivation(i_gate)

        g_delta=state_delta*i_gate*sigmoid_otput_to_derivation(g_gate)




        synapse_h_o_update +=np.atleast_2d(h_output).T.dot(output_delta)


        synapse_U_i_update +=alpha*np.atleast_2d(h_prev_output).T.dot(i_delta)
        synapse_U_f_update += alpha * np.atleast_2d(h_prev_output).T.dot(f_delta)
        synapse_U_o_update += alpha * np.atleast_2d(h_prev_output).T.dot(o_delta)
        synapse_U_new_update+= alpha * np.atleast_2d(h_prev_output).T.dot(g_delta)



        synapse_x_i_update +=alpha*np.atleast_2d(X).T.dot(i_delta)
        synapse_x_f_update += alpha * np.atleast_2d(X).T.dot(f_delta)
        synapse_x_o_update += alpha * np.atleast_2d(X).T.dot(o_delta)
        synapse_x_new_update += alpha * np.atleast_2d(X).T.dot(g_delta)


        future_o_delta =o_delta
        future_i_delta =i_delta
        future_f_delta =f_delta
        future_g_delta =g_delta
        future_state_delta =state_delta
        future_f_gate =f_gate



    synapse_h_o+=synapse_h_o_update

    synapse_U_i+=synapse_U_i_update
    synapse_U_f+=synapse_U_f_update
    synapse_U_o+=synapse_U_o_update
    synapse_U_new+=synapse_U_new_update

    synapse_x_i+=synapse_x_i_update
    synapse_x_f+=synapse_x_f_update
    synapse_x_o+=synapse_x_o_update
    synapse_x_new+=synapse_x_new_update



    synapse_h_o_update*=0

    synapse_U_i_update*=0
    synapse_U_f_update*=0
    synapse_U_o_update*=0
    synapse_U_new_update*=0

    synapse_x_i_update*=0
    synapse_x_f_update*=0
    synapse_x_o_update*=0
    synapse_x_new_update*=0




    if(epoch%1000==0):
        print "Error: "+str(overallError)
        print "Pred: " +str(d)
        print "true: " +str(c)
        out=0

        for index,x in enumerate(reversed(d)):
            out+=x*pow(2,index)

        print str(a_int)+" + "+str(b_int)+" = "+str(out)


