"""
    Author: Amanda Ortega de Castro Ayres
    Created in: September 19, 2019
    Python version: 3.6
"""
from Least_SRMTL import Least_SRMTL
import libmr
from matplotlib import pyplot, cm
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D, art3d
import numpy as np
import numpy.matlib
import sklearn.metrics

class EVeC(object):
    """
    Extreme Value evolving Classifier
    Ruled-based classifier with EVM at the definition of the antecedent of the rules.    
    1. Create a new instance and provide the model parameters;
    2. Call the predict(x) method to make predictions based on the given input;
    3. Call the train(x, y) method to evolve the model based on the new input-output pair.
    """

    # version
    NO_REGRESSION = 0
    LS = 1
    LEAST_SRMTL = 2
    LOGISTIC_SRMTL = 3

    # Model initialization
    def __init__(self, L, sigma=0.5, delta=50, N=np.Inf, version=0, rho=None):
        # Setting EVeC algorithm parameters
        self.L = L
        self.sigma = sigma
        self.tau = 99999
        self.delta = delta
        self.N = N    
        self.version = version
        self.rho = rho

        self.mr_x = list()
        self.x0 = list()        
        self.X = list()
        self.step = list()
        self.last_update = list()
        self.c = 0

        # Version that trains the consequent using linear regression
        if version == EVeC.NO_REGRESSION:
            self.label = list()
        else:
            self.y = list()
            self.theta = list()

            if self.rho is not None:
                self.init_theta = 2
                self.srmtl = Least_SRMTL(rho)
                self.R = None        

    # Initialization of a new instance of rule.
    def add_rule(self, x0, step, label=None, y0=None):
        self.mr_x.append(libmr.MR())
        self.x0.append(x0)        
        self.X.append(x0)
        self.step.append(step)
        self.last_update.append(np.max(step))
        self.c = self.c + 1

        if self.version == 0:
            self.label.append(label)
        else:
            self.y.append(y0.reshape(1, -1))            

            if self.rho is None:
                self.theta.append(np.linalg.lstsq(np.insert(self.X[-1], 0, 1, axis=1), self.y[-1], rcond=None)[0])
            else:
                self.theta.append(np.zeros((x0.shape[0], x0.shape[1] + 1)))
                self.init_theta = 2

    # Add the sample(s) (X, label) as covered by the extreme vector. Remove repeated points.
    def add_sample_to_rule(self, index, X, step, y=None):
        self.X[index] = np.concatenate((self.X[index], X))        
        self.step[index] = np.concatenate((self.step[index], step))          

        if self.version != 0:
            self.y[index] = np.concatenate((self.y[index], y))

        if self.X[index].shape[0] > self.N:
            indexes = np.argsort(-self.step[index].reshape(-1))

            self.X[index] = self.X[index][indexes[: self.N], :]
            self.step[index] = self.step[index][indexes[: self.N]]

            if self.version != 0:
                self.y[index] = self.y[index][indexes[: self.N]]            
        
        self.x0[index] = np.average(self.X[index], axis=0).reshape(1, -1)
        self.last_update[index] = np.max(self.step[index])

        if self.version != 0 and self.rho is None:
                self.theta[index] = np.linalg.lstsq(np.insert(self.X[index], 0, 1, axis=1), self.y[index], rcond=None)[0]        

    def delete_from_list(self, list_, indexes):
        for i in sorted(indexes, reverse=True):
            del list_[i]
        
        return list_

    # Calculate the firing degree of the sample to the psi curve
    def firing_degree(self, index, x):
            return self.mr_x[index].w_score_vector(sklearn.metrics.pairwise.pairwise_distances(self.x0[index], x).reshape(-1))

    # Fit the psi curve of the EVs according to the external samples 
    def fit(self, index, X_ext):
        self.fit_x(index, sklearn.metrics.pairwise.pairwise_distances(self.x0[index], X_ext)[0])

    # Fit the psi curve to the extreme values with distance D to the center of the EV
    def fit_x(self, index, D):
        self.mr_x[index].fit_low(1/2 * D, min(D.shape[0], self.tau))

    # Get the distance from the origin of the input rule which has the given probability to belong to the curve
    def get_distance_input(self, percentage, index=None):
        if index is None:
            return [self.mr_x[i].inv(percentage) for i in range(self.c)]
        else:
            return self.mr_x[index].inv(percentage)          

    # For version zero, obtain the samples that do not belong to the given rule and have the same label; for version one, return all the samples that do not belong to the given rule
    def get_external_samples(self, index=None):
        if index is None:
            return np.concatenate(self.X)
        else:
            if self.version == 0:
                indexes = [i for i in range(len(self.label)) if self.label[i] == self.label[index] and i != index]

                if len(indexes) > 0:
                        return np.concatenate(list(map(self.X.__getitem__, indexes)))

            if self.c > 1:
                # if there is no other rule besides the current rule with the same label, return the remaining rules regardless the label content
                return np.concatenate(self.X[:index] + self.X[index + 1 :])
            
            return np.array([])

    # Merge two rules of different clusters whenever the origin of one is inside the sigma probability of inclusion of the psi curve of the other
    def merge(self):
        self.sort_rules()
        index = 0
        
        while index < self.c:
            if index + 1 < self.c:
                x0 = np.concatenate(self.x0[index + 1 : ])

                S_index = self.firing_degree(index, x0)
                index_to_merge = np.where(S_index > self.sigma)[0] + index + 1

                if index_to_merge.size > 0:
                    self.init_theta = 2                

                for i in reversed(range(len(index_to_merge))):
                    if self.version != 0:
                        self.add_sample_to_rule(index, self.X[index_to_merge[i]], self.step[index_to_merge[i]], self.y[index_to_merge[i]])
                        self.remove_rule([index_to_merge[i]])
                    elif self.label[index] == self.label[index_to_merge[i]]:
                        self.add_sample_to_rule(index, self.X[index_to_merge[i]], self.step[index_to_merge[i]])                        
                        self.remove_rule([index_to_merge[i]])
            
            index = index + 1

    # Plot the granules that form the antecedent part of the rules
    def plot(self, name_figure_input, name_figure_output, step):
        # Input fuzzy granules plot
        fig = pyplot.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.axes.set_xlim3d(left=-2, right=2) 
        ax.axes.set_ylim3d(bottom=-2, top=2) 
        z_bottom = -0.3
        ax.set_zticklabels("")     

        colors = cm.get_cmap('Dark2', self.c)

        for i in range(self.c):
            self.plot_rule_input(i, ax, '.', colors(i), z_bottom)        

        # Plot axis' labels
        ax.set_xlabel('u(t)', fontsize=15)
        ax.set_ylabel('y(t)', fontsize=15)
        ax.set_zlabel('$\mu_x$', fontsize=15)    

        # Save figure
        fig.savefig(name_figure_input)

        # Close plot
        pyplot.close(fig)

    # Plot the probability of sample inclusion (psi-model) together with the samples associated with the rule for the input fuzzy granules
    def plot_rule_input(self, index, ax, marker, color, z_bottom):
        # Plot the input samples in the XY plan
        ax.scatter(self.X[index][:, 0], self.X[index][:, 1], z_bottom * np.ones((self.X[index].shape[0], 1)), marker=marker, color=color)

        # Plot the radius for which there is a probability sigma to belong to the rule
        radius = self.get_distance_input(self.sigma, index)
        p = Circle((self.x0[index][0, 0], self.x0[index][0, 1]), radius, fill=False, color=color)
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=z_bottom, zdir="z")

        # Plot the psi curve of the rule
        r = np.linspace(0, self.get_distance_input(0.05, index), 100)
        theta = np.linspace(0, 2 * np.pi, 145)    
        radius_matrix, theta_matrix = np.meshgrid(r,theta)            
        X = self.x0[index][0, 0] + radius_matrix * np.cos(theta_matrix)
        Y = self.x0[index][0, 1] + radius_matrix * np.sin(theta_matrix)
        points = np.array([np.array([X, Y])[0, :, :].reshape(-1), np.array([X, Y])[1, :, :].reshape(-1)]).T
        Z = self.firing_degree(index, points)
        ax.plot_surface(X, Y, Z.reshape((X.shape[0], X.shape[1])), antialiased=False, cmap=cm.coolwarm, alpha=0.1)

    # Predict the output given the input sample x
    def predict(self, x):
        # there is no rule
        if self.c == 0:
            return np.zeros(self.L, dtype=int)

        if self.version == 0:
            output_labels = np.zeros(self.L, dtype=int)
            
            max_firing = 0
            label_max_firing = -1
            fired = False

            for i in range(self.c):
                firing = self.firing_degree(i, x)

                if firing >= self.sigma:
                    output_labels[self.label[i]] = 1
                    fired = True
                
                if firing > max_firing:
                    max_firing = firing
                    label_max_firing = self.label[i]

            # guarantee that at least one label is set to 1
            if not fired:
                output_labels[label_max_firing] = 1

            return output_labels
        
        output = np.zeros((self.c, self.L))
        firing = np.zeros(self.c)

        for i in range(self.c):
            output[i, :] = self.predict_rule(i, x)
            firing[i] = self.firing_degree(i, x)
        
        # weighted average of the local predictions and the corresponding firing degrees
        output = np.sum(np.multiply(output, np.repeat(firing.reshape(-1, 1), self.L, axis=1)), axis=0) / np.sum(firing)

        # the final values: 1 if above 0.5 and 0 otherwise
        final_output = np.where(output >= 0.5, 1, 0)

        # guarantee that at least one label is set to 1
        if np.all(final_output == 0):            
            final_output[np.unravel_index(np.argmax(output, axis=None), output.shape)] = 1

        return final_output

    # Predict the local output of x based on the linear regression of the samples stored at the rule
    def predict_rule(self, index, x):
        return np.insert(x, 0, 1).reshape(1, -1) @ self.theta[index]

    # Calculate the degree of relationship of all the rules to the rule of index informed as parameter
    def relationship_rules(self, index):
        distance_x = sklearn.metrics.pairwise.pairwise_distances(self.x0[index], np.concatenate(self.x0)).reshape(-1)        
        relationship_x_center = self.mr_x[index].w_score_vector(distance_x)                
        relationship_x_radius = self.mr_x[index].w_score_vector(distance_x - self.get_distance_input(self.sigma))        

        return np.maximum(relationship_x_center, relationship_x_radius) 

    # Remove the rule whose index was informed by parameter
    def remove_rule(self, index):
        self.mr_x = self.delete_from_list(self.mr_x, index)
        self.x0 = self.delete_from_list(self.x0, index)        
        self.X = self.delete_from_list(self.X, index)
        self.step = self.delete_from_list(self.step, index)
        self.last_update = self.delete_from_list(self.last_update, index)
        self.c = len(self.mr_x)

        if self.version == 0:
            self.label = self.delete_from_list(self.label, index)
        else:
            self.y = self.delete_from_list(self.y, index)
            self.theta = self.delete_from_list(self.theta, index)

    # Remove the rules that didn't have any update in the last threshold steps
    def remove_outdated_rules(self, threshold):
        indexes_to_remove = list()

        for index in range(self.c):
            if self.last_update[index] <= threshold:
                indexes_to_remove.append(index)

        if len(indexes_to_remove) > 0:
            self.remove_rule(indexes_to_remove)

            if self.rho is not None:
                self.update_R()
                self.init_theta = 2            

    # Sort the rules according to the last update
    def sort_rules(self):
        new_order = (-np.array(self.last_update)).argsort()

        self.mr_x = list(np.array(self.mr_x)[new_order])
        self.x0 = list(np.array(self.x0)[new_order])        
        self.X = list(np.array(self.X)[new_order])
        self.step = list(np.array(self.step)[new_order])
        self.last_update = list(np.array(self.last_update)[new_order])

        if self.version == 0:
            self.label = list(np.array(self.label)[new_order])
        else:
            self.y = list(np.array(self.y)[new_order])

    # Evolves the model (main method)
    def train(self, x, y, step):
        if self.version == 0:
            best_rule = -1 * np.ones(y.size, dtype=int)
            best_rule_value = np.zeros(y.size)

            # check if it is possible to insert the sample in existing rules
            for index in range(self.c):
                # se a amostra está associada ao rótulo da regra atual 
                if y[self.label[index]]:
                    tau = self.firing_degree(index, x)

                    if tau > best_rule_value[self.label[index]] and tau > self.sigma:
                        best_rule[self.label[index]] = index
                        best_rule_value[self.label[index]] = tau

            for index in range(y.size):
                if y[index]:
                    # Add the sample to an existing rule
                    if best_rule[index] != -1:
                        self.add_sample_to_rule(best_rule[index], x, step)
                    # Create a new rule
                    else:
                        self.add_rule(x, step, label=index)
        else:
            best_EV = None
            best_EV_value = 0            

            # check if it is possible to insert the sample in an existing model
            for index in range(self.c):
                tau = self.firing_degree(index, x)

                if tau > best_EV_value and tau > self.sigma:
                    best_EV = index
                    best_EV_value = tau

            update = False

            # Add the sample to an existing EV
            if best_EV is not None:
                self.add_sample_to_rule(best_EV, x, step, y.reshape(1, -1))
            # Create a new EV
            else:
                self.add_rule(x, step, y0=y)
                update = True
        
        self.update_rules()

        if step != 0 and (step % self.delta) == 0:      
            self.remove_outdated_rules(step[0, 0] - self.delta)
            self.merge()
            update = True

        if self.rho is not None:
            if update:
                self.update_R()

            self.theta = self.srmtl.train(self.X, self.y, self.init_theta)
            self.init_theta = 1            

    # Update the psi curve of the rules
    def update_rules(self):
        for i in range(self.c):
            if self.version == 0:
                X_ext = self.get_external_samples(i)

                if X_ext.shape[0] > 0:
                    self.fit(i, X_ext)
            else:
                X_ext = self.get_external_samples(i)

                if X_ext.shape[0] > 0:
                    self.fit(i, X_ext)                

    def update_R(self):        
        S = np.zeros((self.c, self.c))

        for i in range(self.c):
            S[i, :] = self.relationship_rules(i)

        self.R = None

        for i in range(self.c):
            for j in range(i + 1, self.c):
                if S[i, j] > 0 or S[j, i] > 0:
                    edge = np.zeros((self.c, 1))

                    edge[i] = max(S[i, j], S[j, i])
                    edge[j] = - max(S[i, j], S[j, i])

                    if self.R is None:
                        self.R = edge
                    else:
                        self.R = np.concatenate((self.R, edge), axis=1)
                    
        self.srmtl.set_RRt(self.R)                