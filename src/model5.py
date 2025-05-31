from encoder import *


class Model5(Model):
    def __init__(self, hp, graph):
        super(Model4, self).__init__()
        self.hp = hp
        self.graph = graph

    def prepare(self):
        # Hyperparameters
        self.number_of_epochs = self.hp.get('epochs')
        self.patience = self.hp.get('patience')
        self.split = self.hp.get('split')
        self.batch_size = self.hp.get('batch_size')
        self.regularization_rate = self.hp.get('regularization_rate')
        self.learning_rate = self.hp.get('learning_rate')
        self.dropout_rate = self.hp.get('dropout_rate')
        self.input_layer_dimension = self.hp.get('input_layer_dimension')
        self.layer_1_dimension = self.hp.get('layer_1_dimension')
        self.layer_2_dimension = self.hp.get('layer_2_dimension')
        self.layer_3_dimension = self.hp.get('layer_3_dimension')
        self.output_layer_dimension = self.hp.get('output_layer_dimension')

        self.optimizer = Optimizers.optimizer(self.learning_rate)
        self.train_mask, self.test_mask, self.valid_mask = self.graph.node_labels_sampler(self.split)
        self.graph.y = []
        for n, label in self.graph.nodes.items():
            y_probabilty_vector = list(np.zeros((self.graph.number_of_classes)))
            # y_probabilty_vector[node_indices[label]] = 1.0
            y_probabilty_vector[int(label) - 1] = 1.0
            self.graph.y.append(y_probabilty_vector)
        self.graph.y = tf.reshape(tf.convert_to_tensor(self.graph.y, dtype=tf.float32),
                                  shape=(self.graph.number_of_nodes, self.graph.number_of_classes))
        self.graph.X = tf.Variable(Initializers.identity(
            Shape(in_size=self.graph.number_of_nodes, out_size=self.input_layer_dimension, batch_size=self.batch_size)),
            dtype=tf.float32)
        self.graph.degree_normalized_adjacency = tf.constant(
            tf.convert_to_tensor(self.graph.degree_normalized_adjacency, dtype=tf.float32), dtype=tf.float32)

    def compile(self):
        # Input Dense layer
        self.input_layer_operation_1 = Linear(input=Input(input=self.graph.features, normalize=True),
                                              shape=Shape(in_size=self.graph.dimensions,
                                                          out_size=self.input_layer_dimension,
                                                          batch_size=self.batch_size))
        # First GCN layer
        self.layer_1_operation_1 = DirectionalGraphConvolution(input=self.input_layer_operation_1,
                                                               in_adjacency=self.graph.degree_normalized_weighted_in_adjacency,
                                                               out_adjacency=self.graph.degree_normalized_weighted_out_adjacency,
                                                               shape=Shape(in_size=self.input_layer_dimension,
                                                                           out_size=self.layer_1_dimension,
                                                                           batch_size=self.batch_size))
        self.layer_1_operation_2 = Tanh(input=self.layer_1_operation_1,
                                        shape=Shape(in_size=self.layer_1_dimension, out_size=self.layer_1_dimension,
                                                    batch_size=self.batch_size))
        self.layer_1_operation_3 = Residual(input1=self.layer_1_operation_1, input2=self.layer_1_operation_2,
                                            shape=Shape(in_size=self.layer_1_dimension, out_size=self.layer_1_dimension,
                                                        batch_size=self.batch_size))
        self.layer_1_operation_4 = Regularize(input=self.layer_1_operation_3, rate=self.regularization_rate,
                                              shape=Shape(in_size=self.layer_1_dimension,
                                                          out_size=self.layer_1_dimension,
                                                          batch_size=self.batch_size))
        self.layer_1_operation_5 = Dropout(input=self.layer_1_operation_4,
                                           shape=Shape(in_size=self.layer_1_dimension, out_size=self.layer_1_dimension,
                                                       batch_size=self.batch_size), rate=self.dropout_rate)
        # Second GCN layer
        self.layer_2_operation_1 = DirectionalGraphConvolution(input=self.layer_1_operation_5,
                                                               in_adjacency=self.graph.degree_normalized_weighted_in_adjacency,
                                                               out_adjacency=self.graph.degree_normalized_weighted_out_adjacency,
                                                               shape=Shape(in_size=self.layer_1_dimension,
                                                                           out_size=self.layer_2_dimension,
                                                                           batch_size=self.batch_size))
        self.layer_2_operation_2 = Tanh(input=self.layer_2_operation_1,
                                        shape=Shape(in_size=self.layer_2_dimension, out_size=self.layer_2_dimension,
                                                    batch_size=self.batch_size))
        self.layer_2_operation_3 = Residual(input1=self.layer_2_operation_1, input2=self.layer_2_operation_2,
                                            shape=Shape(in_size=self.layer_2_dimension, out_size=self.layer_2_dimension,
                                                        batch_size=self.batch_size))
        self.layer_2_operation_4 = Dropout(input=self.layer_2_operation_3,
                                           shape=Shape(in_size=self.layer_2_dimension, out_size=self.layer_2_dimension,
                                                       batch_size=self.batch_size), rate=self.dropout_rate)
        # Third GCN layer
        self.layer_3_operation_1 = DirectionalGraphConvolution(input=self.layer_2_operation_4,
                                                               in_adjacency=self.graph.degree_normalized_weighted_in_adjacency,
                                                               out_adjacency=self.graph.degree_normalized_weighted_out_adjacency,
                                                               shape=Shape(in_size=self.layer_2_dimension,
                                                                           out_size=self.layer_3_dimension,
                                                                           batch_size=self.batch_size))
        self.layer_3_operation_2 = Tanh(input=self.layer_3_operation_1,
                                        shape=Shape(in_size=self.layer_3_dimension, out_size=self.layer_3_dimension,
                                                    batch_size=self.batch_size))
        self.layer_3_operation_3 = Residual(input1=self.layer_3_operation_1, input2=self.layer_3_operation_2,
                                            shape=Shape(in_size=self.layer_3_dimension, out_size=self.layer_3_dimension,
                                                        batch_size=self.batch_size))
        self.layer_3_operation_4 = Normalize(input=self.layer_3_operation_3,
                                             shape=Shape(in_size=self.layer_3_dimension,
                                                         out_size=self.layer_3_dimension,
                                                         batch_size=self.batch_size))
        self.layer_3_operation_5 = Dropout(input=self.layer_3_operation_4,
                                           shape=Shape(in_size=self.layer_3_dimension, out_size=self.layer_3_dimension,
                                                       batch_size=self.batch_size), rate=self.dropout_rate)
        # Output prediction layer
        self.layer_4_operation_1 = Linear(input=self.layer_3_operation_5,
                                          shape=Shape(in_size=self.layer_3_dimension,
                                                      out_size=self.output_layer_dimension,
                                                      batch_size=self.batch_size))

    def compute(self):
        self.input_layer_operation_1.compute(self.parameters[0])
        self.layer_1_operation_1.compute(self.parameters[1])
        self.layer_1_operation_2.compute()
        self.layer_1_operation_3.compute(self.parameters[2])
        self.layer_1_operation_4.compute()
        self.layer_1_operation_5.compute()
        self.layer_2_operation_1.compute(self.parameters[3])
        self.layer_2_operation_2.compute()
        self.layer_2_operation_3.compute(self.parameters[4])
        self.layer_2_operation_4.compute()
        self.layer_3_operation_1.compute(self.parameters[5])
        self.layer_3_operation_2.compute()
        self.layer_3_operation_3.compute(self.parameters[6])
        self.layer_3_operation_4.compute()
        self.layer_3_operation_5.compute()
        self.layer_4_operation_1.compute(self.parameters[7])

        self.predictions = tf.convert_to_tensor(self.layer_4_operation_1.output)
        self.regularization_constant = self.layer_1_operation_4.regularization_constant
        self.embedding = self.layer_3_operation_4.embedding

    def evaluate(self, mask):
        self.loss = tf.add(LossFunctions.masked_cross_entropy_loss_evaluater_2(self.predictions, self.graph.y, mask),
                           self.regularization_constant)
        self.accuracy = Evaluation.masked_accuracy_evaluater(self.predictions, self.graph.y, mask)

    def collect(self):
        self.parameters = []
        self.parameters.append(self.input_layer_operation_1.collect())
        self.parameters.append(self.layer_1_operation_1.collect())
        self.parameters.append(self.layer_1_operation_3.collect())
        self.parameters.append(self.layer_2_operation_1.collect())
        self.parameters.append(self.layer_2_operation_3.collect())
        self.parameters.append(self.layer_3_operation_1.collect())
        self.parameters.append(self.layer_3_operation_3.collect())
        self.parameters.append(self.layer_4_operation_1.collect())

    def update(self, tape):
        gradients = tape.gradient(self.train_loss, self.parameters)
        info = {}
        for i, p in enumerate(self.parameters):
            info[i] = len(p)
        self.parameters = [element for innerList in self.parameters for element in innerList]
        gradients = [element for innerList in gradients for element in innerList]
        combined_params_gradients = list(zip(gradients, self.parameters))
        self.optimizer.apply_gradients(combined_params_gradients)
        updated_parameters = []
        for k, v in info.items():
            popped_elements = [self.parameters.pop(0) for _ in range(v)]
            updated_parameters.append(popped_elements)
        self.parameters = updated_parameters

    def update_gpu(self, tape):
        # Gemini converted
        # Keep a reference to the original nested structure (contains tf.Variables)
        original_nested_structure = self.parameters
        # 1. Compute gradients (GPU operation).
        #    `gradients_nested` will have the same nested structure as `original_nested_structure`.
        gradients_nested = tape.gradient(self.train_loss, original_nested_structure)
        if gradients_nested is None:
            print("Warning: tape.gradient returned None for all parameters. Skipping update.")
            return
        # 2. Flatten parameters and gradients (CPU operations, using tf.nest)
        flat_parameters = tf.nest.flatten(original_nested_structure)
        flat_gradients = tf.nest.flatten(gradients_nested)
        # 3. Filter out (None gradient, variable) pairs and handle IndexedSlices
        valid_grads_and_vars = []
        for g, v in zip(flat_gradients, flat_parameters):
            if g is not None:
                if isinstance(g, tf.IndexedSlices):
                    # Convert IndexedSlices to dense tensors if optimizer requires
                    g = tf.convert_to_tensor(g)
                valid_grads_and_vars.append((g, v))
        if not valid_grads_and_vars:
            print("Warning: No valid gradients to apply after filtering. Skipping optimizer step.")
            return
        # 4. Apply gradients (GPU operation)
        #    This updates the tf.Variable objects in `valid_grads_and_vars` (and thus
        #    also within `original_nested_structure`) in-place.

        self.optimizer.apply_gradients(valid_grads_and_vars)

        # 5. Reconstruct self.parameters if necessary (CPU operation)
        #    The tf.Variable objects within `original_nested_structure` are already updated.
        #    If `self.parameters` needs to be a *new* list object with this structure
        #    (e.g., if other code expects a fresh list object after each update, though unusual),
        #    or if `original_nested_structure` was a temporary copy, you can repack.
        #    Otherwise, `original_nested_structure` itself already reflects the updates.
        #    For consistency with your original code's intent of rebinding `self.parameters`:
        self.parameters = tf.nest.pack_sequence_as(
            structure=original_nested_structure,  # The original structure template
            flat_sequence=flat_parameters  # The flat list of (now updated) variables
        )
        # Note: `flat_parameters` contains the variable objects that were updated.
        # `pack_sequence_as` will build a new nested structure containing these same variable objects.

    #@tf.function
    def train(self):
        # Visualization
        self.losses = []
        self.accuracies = []
        self.epochs = []
        for epoch in tf.range(self.number_of_epochs):
            # Timing
            start_time = time.perf_counter()
            # Train
            with tf.GradientTape() as tape:
                self.collect()
                self.compute()
                self.evaluate(self.train_mask)
                self.train_loss = self.loss
                self.train_accuracy = self.accuracy
            self.update(tape)
            # self.update_gpu(tape)
            # Test
            self.collect()
            self.compute()
            self.evaluate(self.test_mask)
            self.test_loss = self.loss
            self.test_accuracy = self.accuracy
            # Timing
            end_time = time.perf_counter()
            # Print and Visualize Information
            self.time_per_epoch = tf.constant(round((end_time - start_time), 3), dtype=tf.float32)
            tf.print(" Epoch: " + tf.strings.as_string(epoch) + " Seconds/Epoch: " + tf.strings.as_string(
                self.time_per_epoch) + " Learning Rate: " + tf.strings.as_string(
                tf.constant(round(self.learning_rate, 3), dtype=tf.float32)) + " Train Loss: " + tf.strings.as_string(
                self.train_loss) + " Train Accuracy: " + tf.strings.as_string(
                self.train_accuracy) + " Test Loss: " + tf.strings.as_string(
                self.test_loss) + " Test Accuracy: " + tf.strings.as_string(self.test_accuracy))
            self.losses.append(self.test_loss.numpy())
            self.accuracies.append(self.test_accuracy.numpy())
            self.epochs.append(epoch.numpy())

            # self.losses.append(self.test_loss)
            # self.accuracies.append(self.test_accuracy)
            # self.epochs.append(epoch)

    def validate(self):
        self.collect()
        self.compute()
        self.evaluate(self.valid_mask)
        self.valid_loss = self.loss
        self.valid_accuracy = self.accuracy
