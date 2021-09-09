import numpy as np
from scipy import stats
from scipy.special import expit as sigmoid
import statsmodels.api as sm
import os.path


class NeuralMassModel:
    """
    Refactored code for running the neural mass simulation from:
     Cole MW, Ito T, Schultz D, Mill R, Chen R, Cocuzza C (2019).
     "Task activations produce spurious but systematic inflation of task functional connectivity estimates".
     NeuroImage. doi:10.1016/j.neuroimage.2018.12.054
     https://github.com/ColeLab/TaskFCRemoveMeanActivity/blob/master/neuralmassmodel/NeuralMassModel.ipynb
    """

    def __init__(self, num_regions, num_modules=3, num_regions_per_modules=None, struct_conn_probs=None,
                 syn_com_mult=None, syn_weight_std=0.001):
        """

        Args:
            num_regions (int): number of interacted regions
            num_modules (int): number of modules in structural connection
            struct_conn_probs (dict): dictionary with structural connections probabilities between the modules
            num_regions_per_module (list of int): len should be equal to num_modules, and sum of regions should be equal to num_regions
            syn_com_mult (dict): dictionary with synaptic weights to existing structural connections
            syn_weight_std (float): standard deviation for synaptic weight
            g (float):
            indep (float):
        """
        self.num_regions = num_regions
        self.num_modules = num_modules
        self.syn_weight_std = syn_weight_std
        if struct_conn_probs is None:
            self.struct_conn_probs = {'in': 0.8, 'out': 0.2}
        else:
            self.struct_conn_probs = struct_conn_probs

        if num_regions_per_modules is None:
            num_equal = int(round(num_regions / num_modules))
            self.num_regions_per_modules = (num_modules - 1) * [num_equal] + [
                num_regions - (num_modules - 1) * num_equal]
        else:
            self.num_regions_per_modules = num_regions_per_modules
        assert num_regions == np.sum(
            self.num_regions_per_modules), "Sum number in each regions (num_regions_per_modules) should be equal to num_regions"
        self.module_borders = [0] + list(np.cumsum(self.num_regions_per_modules))

        self._init_struct_matrix()

        if syn_com_mult is None:
            self.syn_com_mult = {0: [1, 1, 0], 1: [1, 1, 0], 2: [0, 0, 1]}
        else:
            self.syn_com_mult = syn_com_mult

        assert len(self.syn_com_mult.keys()) == self.num_modules, "Number of keys should corresponds to number of " \
                                                                  "modules "
        self._init_synaptic_weights()

    def _init_struct_matrix(self):
        self.struct_matrix = np.random.uniform(0, 1, (self.num_regions, self.num_regions)) > 1 - self.struct_conn_probs[
            'out']
        for i in range(self.num_modules):
            self.struct_matrix[self.module_borders[i]:self.module_borders[i + 1],
            self.module_borders[i]:self.module_borders[i + 1]] = \
                np.random.uniform(0, 1, (self.num_regions_per_modules[i], self.num_regions_per_modules[i])) > 1 - \
                self.struct_conn_probs['in']

        np.fill_diagonal(self.struct_matrix, 1)

    def _init_synaptic_weights(self):
        #todo check synaptic
        self.synaptic_weight = self.struct_matrix * (
                    1 + np.random.standard_normal((self.num_regions, self.num_regions)) * self.syn_weight_std)
        for row in range(self.num_modules):
            for col in range(self.num_modules):
                self.synaptic_weight[self.module_borders[row]:self.module_borders[row + 1],
                self.module_borders[col]:self.module_borders[col + 1]] *= self.syn_com_mult[row][col]
        np.fill_diagonal(self.synaptic_weight, 0)
        for node in range(self.num_regions):
            k = np.sum(self.synaptic_weight[node, :])
            if k > 0:
                self.synaptic_weight[node, :] = np.divide(self.synaptic_weight[node, :], k)

    def _init_synpatic_cole(self):
        struct_conn_vector = np.random.uniform(0, 1, (self.num_regions, self.num_regions)) > .90
        # Add self-connections (important if adding autocorrelation later)
        np.fill_diagonal(struct_conn_vector, 10)
        self.struct_matrix = struct_conn_vector
        # Create modular structural network (3 modules)
        num_modules = 3
        numr_per_module = int(round(self.num_regions / num_modules))
        lastModuleNode = -1
        for moduleNum in range(0, num_modules):
            for thisNodeNum in range(lastModuleNode + 1, lastModuleNode + numr_per_module + 1):
                # Set this node to connect to 10 random other nodes in module
                for i in range(1, numr_per_module // 2):
                    randNodeInModule = int(
                        np.random.uniform(lastModuleNode + 1, lastModuleNode + numr_per_module + 1, (1, 1)))
                    struct_conn_vector[thisNodeNum, randNodeInModule] = 1
            lastModuleNode = lastModuleNode + numr_per_module

        # Adding synaptic weights to existing structural connections (small random synapse strength variation)
        synaptic_weight_vector = struct_conn_vector * (1 + np.random.standard_normal((self.num_regions, self.num_regions)) * .001)

        # Adding synaptic mini-communities (within community 1)
        synaptic_weight_vector[0:50, 50:100] = synaptic_weight_vector[0:50, 50:100] * -0.2
        synaptic_weight_vector[50:100, 0:50] = synaptic_weight_vector[50:100, 0:50] * -0.2
        synaptic_weight_vector[0:50, 0:50] = synaptic_weight_vector[0:50, 0:50] * 1.2
        synaptic_weight_vector[50:100, 50:100] = synaptic_weight_vector[50:100, 50:100] * 1.2

        # MODIFICATION: 0 connectivity between structural community 1 and 3
        synaptic_weight_vector[0:100, 200:300] = synaptic_weight_vector[0:100, 200:300] * 0
        synaptic_weight_vector[200:300, 0:100] = synaptic_weight_vector[200:300, 0:100] * 0
        synaptic_weight_vector[100:200, 200:300] = synaptic_weight_vector[100:200, 200:300] * 0
        synaptic_weight_vector[200:300, 100:200] = synaptic_weight_vector[200:300, 100:200] * 0

        # Normalize each region's inputs to have a mean of 1/k, where k is the number of incoming connections (in degree)
        # Based on Barral, J. & Reyes, A. D. Synaptic scaling rule preserves excitatory-inhibitory balance and salient neuronal network dynamics. Nat. Neurosci. (2016)
        # This ensures that all inputs into each node sum to 1
        for nodeNum in range(self.num_regions):
            k = np.sum(synaptic_weight_vector[nodeNum, :])
            if k > 0:
                synaptic_weight_vector[nodeNum, :] = np.divide(synaptic_weight_vector[nodeNum, :], k)

        self.synaptic_weight  = synaptic_weight_vector


    def compute_network_model_cole(self, num_time_points, bias_param=-20, spont_act_level=3,
                              g=5.0, indep=1.0, k=1):

        outputdata = {'input_activity': np.zeros((num_time_points, self.num_regions)),
                  'output_activity': np.zeros((num_time_points, self.num_regions))}

        bias = np.zeros(shape=(self.num_regions,))
        bias[range(self.num_regions)] = bias_param
        autocorr = 0.0
        global_coupling_mat = self.synaptic_weight*g
        np.fill_diagonal(global_coupling_mat, 0)
        indep_var_mat = np.identity(self.num_regions)*indep
        self.synaptic_weight = global_coupling_mat+indep_var_mat
        outputvect = np.zeros(self.num_regions)

        for this_time_point in range(0, num_time_points):
            stim_act_vector = np.zeros(self.num_regions)
            if this_time_point == 0:
                # Generate spontaneous activity for initial state
                act_vector = sigmoid(k * (bias + np.random.normal(0, spont_act_level, (self.num_regions,))))
                input_activity = np.zeros(self.num_regions)
            else:
                # Bring outputs from previous time point as inputs for this timepoint
                act_vector = outputvect

            act_vector_to_mat = np.kron(np.ones((self.num_regions, 1)), act_vector)
            input_activity = np.multiply(autocorr, input_activity) + np.sum(
                np.multiply(act_vector_to_mat, self.synaptic_weight), axis=1)

            input_activity = input_activity + np.random.normal(0, spont_act_level, (self.num_regions,))
            outputvect = sigmoid(k * (bias + input_activity))
            outputdata['input_activity'][this_time_point, :] = input_activity
            outputdata['output_activity'][this_time_point, :] = outputvect
        return outputdata

    def compute_network_model(self, num_time_points, bias_param=-20, spont_act_level=3,
                              g=5.0, indep=1.0, stim_times=None, stim_t_par_start=None, stim_t_par_end=None,
                              stim_r_par_start=None, stim_r_par_end=None, stim_regions=None, stim_param=3., k=1):
        """
            For each region i
            output_activity_i (t) = f( input_activity_i (t) + bias )
            where output_activity_i (t) is the output activity (population spike rate) for unit i at time t,
            input_activity_i is the input (population field potential) as defined below,
            and Bias is the bias (population resting potential, or excitability).

            input_activity_i (t) = \sum_{j=1}^n G w_{ij} output_activity_j (t-1) + spont_activity_i + stim_activity_i
            where input_activity(t) is the input (population field potential) for unit i at time t,
             G is the global coupling parameter (a scalar influencing all connection strengths),
             w_{ij} is the synaptic weight from unit j to i, output_activity_{j}(t-1) is the output activity
             from unit j at the previous timestep, spont_activity_i is spontaneous activity (a Gaussian random value),
             and stim_activity is task stimulation (if any).


        Args:
            stim_param (float): activation parameter in ooriginal source equal to 0.3
            num_time_points (int): number of time points to simulate
            bias_param (float): shifts all inputs to regions, with negative values making reaching "threshold" less likely
                                Equivalent to aggregate resting potentials of neurons in population
            spont_act_level: spontaneous activity parameter for sigma in normal distribution
            g (float):  positive scalar, the global coupling parameter (a scalar influencing all connection strengths)
            indep (float): parameter modulated strength of previous state influence to current
            stim_times (list of time moments): time moment where regions were stimulated
            stim_regions (list of stimulated regions): list of stimulated regions
            k: sigmoid multiplicator parameter

        Returns:

        """
        assert (stim_times is None) or (stim_t_par_start is None), "Couldn't be both not None"
        assert (stim_regions is None) or (stim_r_par_start is None), "Couldn't be both not None"

        self.synaptic_weight = g * self.synaptic_weight
        np.fill_diagonal(self.synaptic_weight, indep)
        input_activity = np.zeros(shape=(self.num_regions, num_time_points))
        bias = bias_param * np.ones(shape=(self.num_regions,))
        stim_activity = np.zeros(shape=(self.num_regions, num_time_points))
        if (stim_times is not None) and (stim_regions is not None):
            l = [(r, t) for r in stim_regions for t in stim_times]
            row, col = list(zip(*l))
            stim_activity[row, col] = stim_param
        if (stim_t_par_start is not None) and (stim_r_par_start is not None):
            assert stim_t_par_end is not None, "End time moment should be defined"
            assert stim_r_par_end is not None, "End region number should be defined"
            stim_activity[stim_r_par_start:stim_r_par_end, stim_t_par_start:stim_t_par_end] = stim_param

        spont_activity = np.random.normal(0, spont_act_level, (self.num_regions, num_time_points))

        output_activity = np.zeros(shape=(self.num_regions, num_time_points))
        output_activity[:, 0] = sigmoid(input_activity[:, 0] + bias + spont_activity[:, 0])
        for i in range(1, num_time_points):
            input_activity[:, i] = self.synaptic_weight @ output_activity[:, i - 1] + spont_activity[:, i] + stim_activity[:, i]
            output_activity[:, i] = sigmoid(k*(input_activity[:, i] + bias))
        return {'input_activity': input_activity, 'output_activity': output_activity}