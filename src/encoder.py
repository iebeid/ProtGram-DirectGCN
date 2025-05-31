from grapher import *


class Operation(object):
    def __init__(self, operation: str, shape: Shape):
        print(str(operation) + ":" + str(shape))


class Trainable(Operation):
    def __init__(self, operation: str, shape: Shape):
        super(Trainable, self).__init__("Trainable:" + operation, shape)


class NonTrainable(Operation):
    def __init__(self, operation: str, shape: Shape):
        super(NonTrainable, self).__init__("NonTrainable:" + operation, shape)


class Activation(NonTrainable):
    def __init__(self, operation: str, shape: Shape):
        super(Activation, self).__init__("Activation:" + operation, shape)


class Tanh(Activation):
    def __init__(self, input, shape: Shape):
        super(Tanh, self).__init__("Tanh", shape)
        self.input = input
        self.shape = shape
        self.output = tf.Variable(Initializers.glorot(self.shape), trainable=True, name="Activation:Tanh:output")

    def compute(self):
        self.output = tf.nn.tanh(self.input.output)


class Relu(Activation):
    def __init__(self, input, shape: Shape):
        super(Relu, self).__init__("Relu", shape)
        self.input = input
        self.shape = shape
        self.output = tf.Variable(Initializers.glorot(self.shape), trainable=True, name="Activation:Relu:output")

    def compute(self):
        self.output = tf.nn.relu(self.input.output)


class Softmax(Activation):
    def __init__(self, input, shape: Shape):
        super(Softmax, self).__init__("Softmax", shape)
        self.input = input
        self.shape = shape
        self.output = tf.Variable(Initializers.glorot(self.shape), trainable=True, name="Activation:Softmax:output")

    def compute(self):
        self.output = tf.map_fn(fn=lambda i: tf.nn.softmax(i), elems=self.input)


class Normalize(NonTrainable):
    def __init__(self, input, shape: Shape):
        super(Normalize, self).__init__("Normalize", shape)
        self.input = input
        self.shape = shape
        self.output = tf.Variable(Initializers.glorot(self.shape), trainable=True, name="Activation:Normalize:output")

    def compute(self):
        self.embedding = tf.nn.l2_normalize(self.input.output, axis=1)
        self.output = self.input.output


class Dropout(NonTrainable):
    def __init__(self, input, shape: Shape, rate: float):
        super(Dropout, self).__init__("Dropout", shape)
        self.input = input
        self.shape = shape
        self.output = tf.Variable(Initializers.glorot(self.shape), trainable=True, name="Dropout:output")
        self.rate = rate

    def compute(self):
        self.output = tf.nn.dropout(self.input.output, self.rate)


class Regularize(NonTrainable):
    def __init__(self, input, shape: Shape, rate: float):
        super(Regularize, self).__init__("Regularize", shape)
        self.input = input
        self.shape = shape
        self.output = tf.Variable(Initializers.glorot(self.shape), trainable=True, name="Regularize:output")
        self.regularization_constant = None
        self.rate = rate

    def compute(self):
        r = tf.keras.regularizers.L2(self.rate)
        r_constant = r(self.input.output)
        self.regularization_constant = r_constant
        self.output = self.input.output


class Aggregate(NonTrainable):
    def __init__(self, input, ids, shape: Shape):
        super(Aggregate, self).__init__("Aggregate", shape)
        self.input = input
        self.shape = shape
        self.output = tf.Variable(Initializers.glorot(self.shape), trainable=True, name="Aggregate:output")
        self.ids = tf.convert_to_tensor(ids, dtype=tf.int32)

    def compute(self):
        all_embeddings = []
        for id in self.ids:
            el = tf.reduce_mean(tf.nn.embedding_lookup(self.input.output, tf.convert_to_tensor(id, dtype=tf.int32)),
                                axis=0)
            all_embeddings.append(el)
        all_embedding = tf.convert_to_tensor(all_embeddings, dtype=tf.float32)
        self.output = all_embedding


class Concatenate(NonTrainable):
    def __init__(self, input1, input2, shape: Shape):
        super(Concatenate, self).__init__("Concatenate", shape)
        self.input1 = input1
        self.input2 = input2
        self.shape = shape
        self.output = tf.Variable(Initializers.glorot(self.shape), trainable=True, name="Concatenate:output")

        if self.input1.shape.out_size != self.input2.shape.out_size:
            raise ValueError('Input and output sizes do not match')

    def compute(self):
        self.output = tf.concat([self.input1.output, self.input2.output], axis=0)


class Split(NonTrainable):
    def __init__(self, input, begin1, end1, begin2, end2, shape: Shape):
        super(Split, self).__init__("Split", shape)
        self.input = input
        self.shape = shape
        self.begin1 = begin1
        self.end1 = end1
        self.begin2 = begin2
        self.end2 = end2
        self.output = tf.Variable(Initializers.glorot(self.shape, axis=1), trainable=False, name="Split:output")
        self.result_set_1 = tf.Variable(Initializers.glorot(
            Shape(in_size=int(self.end1[1] + self.begin1[1]), out_size=int(self.end1[1] + self.begin1[1]),
                  batch_size=int(self.end1[0])), axis=1), trainable=False, name="Split:result_set_1")
        self.result_set_2 = tf.Variable(Initializers.glorot(
            Shape(in_size=int(self.end2[1] + self.begin2[1]), out_size=int(self.end2[1] + self.begin2[1]),
                  batch_size=int(self.end2[0])), axis=1), trainable=False, name="Split:result_set_2")

    def compute(self):
        self.result_set_1 = tf.Variable(tf.slice(self.input.output, self.begin1, self.end1), trainable=False,
                                        name="Split:result_set_1")
        self.result_set_2 = tf.Variable(tf.slice(self.input.output, self.begin2, self.end2), trainable=False,
                                        name="Split:result_set_2")
        self.output = self.input

class AveragePooling(NonTrainable):
    def __init__(self, input, shape: Shape):
        super(AveragePooling, self).__init__("AveragePooling", shape)
        self.input = input
        self.shape = shape
        self.output = tf.Variable(Initializers.glorot(self.shape), trainable=True, name="AveragePooling:output")

    def compute(self):
        embedding_layer = tf.keras.layers.Embedding(
            input_dim=self.shape.in_size,  # Vocabulary size from the tensor
            output_dim=self.shape.out_size,  # Embedding dimension from the tensor
            embeddings_initializer=tf.keras.initializers.Constant(self.input.output),
            trainable=False  # Set to True if you want to fine-tune these embeddings during training
        )
        self.output = tf.reduce_mean(embedding_layer, axis=1)

class AttentionPooling(Trainable):
    def __init__(self, input, shape: Shape):
        super(AttentionPooling, self).__init__("AttentionPooling", shape)
        # the input should be a 2 dimensional tensor of shape (batch_size, sequence_length).
        # the batch size is the number of nodes in the graph. and the sequence length is the number
        # of amino acids in each sequences padded to the max.
        self.input = input
        self.shape = shape
        self.output = tf.Variable(Initializers.glorot(self.shape), trainable=True, name="AttentionPooling:output")
        self.W = tf.Variable(Initializers.glorot(
            Shape(in_size=self.shape.in_size, out_size=1, batch_size=self.shape.batch_size), axis=0),
            trainable=True, name="AttentionPooling:W")
        self.initial_embedding_values = tf.random.uniform(shape=(self.shape.batch_size, self.shape.out_size),
                                                     dtype=tf.float32)

    def reference(self):
        import tensorflow as tf
        import numpy as np

        # --- Configuration ---
        vocab_size = 1000  # Number of unique tokens in our vocabulary
        embedding_dim = 64  # The size of the embedding vector for each token
        max_sequence_length = 20  # The maximum length of input sequences
        batch_size = 32  # How many sequences we process at once

        initial_embedding_values = tf.random.uniform(shape=(vocab_size, embedding_dim), dtype=tf.float32)

        # --- 1. Embedding Layer ---
        # This layer will take integer indices and return dense vectors (embeddings)
        # embedding_layer = tf.keras.layers.Embedding(
        #     input_dim=vocab_size,
        #     output_dim=embedding_dim,
        #     input_length=max_sequence_length # Optional, but good practice for fixed-size sequences
        # )

        embedding_layer = tf.keras.layers.Embedding(
            input_dim=initial_embedding_values.shape[0],  # Vocabulary size from the tensor
            output_dim=initial_embedding_values.shape[1],  # Embedding dimension from the tensor
            embeddings_initializer=tf.keras.initializers.Constant(initial_embedding_values),
            trainable=False  # Set to True if you want to fine-tune these embeddings during training
        )

        # --- Sample Input Data ---
        # Let's create a batch of dummy integer sequences.
        # These integers represent token IDs from our vocabulary.
        # Shape: (batch_size, max_sequence_length)
        sample_input_sequences = np.random.randint(
            low=0,
            high=vocab_size,
            size=(batch_size, max_sequence_length)
        )

        # Perform the embedding lookup
        # Output shape: (batch_size, max_sequence_length, embedding_dim)
        embedded_sequences = embedding_layer(sample_input_sequences)

        print("Sample Input Sequences shape:", sample_input_sequences.shape)
        print("Embedded Sequences shape:", embedded_sequences.shape)

        # --- 2. Simple Additive Attention Mechanism ---
        # We'll create a simple mechanism to learn attention weights.
        # For pooling, we typically want attention over the sequence dimension.
        # A common way is to use a dense layer to compute scores for each token,
        # and then a softmax to turn scores into weights that sum to 1.

        # This layer will compute a score for each token's embedding
        attention_scores_layer = tf.keras.layers.Dense(1, use_bias=False)

        # Compute raw attention scores for each token in each sequence
        # Output shape: (batch_size, max_sequence_length, 1)
        raw_attention_scores = attention_scores_layer(embedded_sequences)

        # Remove the last dimension to get shape (batch_size, max_sequence_length)
        raw_attention_scores = tf.squeeze(raw_attention_scores, axis=-1)

        # Apply softmax to get attention weights that sum to 1 across the sequence length
        # Output shape: (batch_size, max_sequence_length)
        attention_weights = tf.nn.softmax(raw_attention_scores, axis=-1)

        print("\nRaw Attention Scores shape:", raw_attention_scores.shape)
        print("Attention Weights shape:", attention_weights.shape)
        print("Sample Attention Weights for first sequence:", attention_weights[0, :].numpy())
        print("Sum of Attention Weights for first sequence:", tf.reduce_sum(attention_weights[0, :]).numpy())

        # --- 3. Attention Pooling ---
        # Now, we use the attention weights to compute a weighted sum of the embeddings.
        # We need to multiply the attention weights (batch_size, sequence_length)
        # with the embedded sequences (batch_size, sequence_length, embedding_dim).
        # For element-wise multiplication to work correctly, we need to add a dimension
        # to the attention weights to match the embedding_dim dimension.

        # Reshape attention weights to (batch_size, max_sequence_length, 1)
        attention_weights_reshaped = tf.expand_dims(attention_weights, axis=-1)

        # Multiply the embeddings by their corresponding attention weights
        # This weights each embedding vector by its importance
        # Output shape: (batch_size, max_sequence_length, embedding_dim)
        weighted_embeddings = embedded_sequences * attention_weights_reshaped

        # Sum across the sequence length dimension to get the final pooled representation
        # Output shape: (batch_size, embedding_dim)
        attention_pooled_output = tf.reduce_sum(weighted_embeddings, axis=1)

        print("\nAttention Weights Reshaped shape:", attention_weights_reshaped.shape)
        print("Weighted Embeddings shape:", weighted_embeddings.shape)
        print("Attention Pooled Output shape:", attention_pooled_output.shape)

        print("\nExample of the final pooled output for the first sequence:")
        print(attention_pooled_output[0, :].numpy())

    def compute(self):
        embedding_layer = tf.keras.layers.Embedding(
            input_dim=self.initial_embedding_values.shape[0],
            output_dim=self.initial_embedding_values.shape[1],
            embeddings_initializer=tf.keras.initializers.Constant(self.initial_embedding_values),
            trainable=False
        )
        embedded_sequences = embedding_layer(self.input.output)
        attention_layer = tf.keras.layers.Embedding(
            input_dim=self.W.shape[0],
            output_dim=self.W.shape[1],
            embeddings_initializer=tf.keras.initializers.Constant(self.W),
            trainable=False
        )
        attention_scores = attention_layer(self.input.output)
        raw_attention_scores = tf.squeeze(attention_scores, axis=-1)
        attention_weights = tf.nn.softmax(raw_attention_scores, axis=-1)
        attention_weights_reshaped = tf.expand_dims(attention_weights, axis=-1)
        weighted_embeddings = embedded_sequences * attention_weights_reshaped
        self.output = tf.reduce_sum(weighted_embeddings, axis=1)



class LayerNorm(Trainable):
    def __init__(self, input, shape: Shape, rate: float):
        super(LayerNorm, self).__init__("LayerNorm", shape)
        self.input = input
        self.shape = shape
        self.output = tf.Variable(Initializers.glorot(self.shape), trainable=True, name="LayerNorm:output")
        self.epsilon = rate
        self.scale = tf.Variable(tf.ones([1, self.shape.out_size]), trainable=True, name="LayerNorm:scale")
        self.beta = tf.Variable(tf.zeros([1, self.shape.out_size]), trainable=True, name="LayerNorm:beta")

    def collect(self):
        return [self.scale, self.beta]

    def compute(self, parameters):
        mean, var = tf.nn.moments(self.input.output, [0])
        operation1 = self.input.output - mean
        operation2 = tf.sqrt(var + self.epsilon)
        operation3 = operation1 / operation2
        operation4 = operation3 * parameters[0]
        operation5 = operation4 + parameters[1]
        self.output = operation5


class Residual(Trainable):
    def __init__(self, input1, input2, shape: Shape):
        super(Residual, self).__init__("Residual", shape)
        self.input1 = input1
        self.input2 = input2
        self.shape = shape
        self.output = tf.Variable(Initializers.glorot(self.shape), trainable=True, name="Residual:output")

        self.M = tf.Variable(Initializers.glorot(
            Shape(in_size=self.shape.in_size, out_size=self.shape.out_size, batch_size=self.shape.batch_size), axis=0),
            trainable=True, name="Residual:M")
        self.b = tf.Variable(Initializers.glorot(
            Shape(in_size=self.shape.batch_size, out_size=self.shape.out_size, batch_size=self.shape.batch_size),
            axis=1), trainable=True, name="Residual:b")

    def collect(self):
        return [self.M, self.b]

    def compute(self, parameters):
        term1 = tf.matmul(self.input1.output, parameters[0])
        term2 = tf.add(term1, parameters[1])
        term3 = tf.add(self.input2.output, term2)
        self.output = term3


class Linear(Trainable):
    def __init__(self, input, shape: Shape):
        super(Linear, self).__init__("Linear", shape)
        self.input = input
        self.shape = shape
        self.output = tf.Variable(Initializers.glorot(self.shape), trainable=True)
        self.W = tf.Variable(Initializers.glorot(
            Shape(in_size=self.shape.in_size, out_size=self.shape.out_size, batch_size=self.shape.batch_size), axis=0),
            trainable=True, name="Linear:W")
        self.b = tf.Variable(Initializers.glorot(
            Shape(in_size=self.shape.batch_size, out_size=self.shape.out_size, batch_size=self.shape.batch_size),
            axis=1), trainable=True, name="Linear:b")

    def collect(self):
        return [self.W, self.b]

    def compute(self, parameters):
        self.output = tf.add(tf.matmul(self.input.output, parameters[0]), parameters[1])


class GraphConvolution(Trainable):
    def __init__(self, input, adjacency, shape: Shape):
        super(GraphConvolution, self).__init__("GraphConvolution", shape)
        self.input = input
        self.shape = shape
        self.output = tf.Variable(Initializers.glorot(self.shape), trainable=True)
        self.adjacency = adjacency

        self.W = tf.Variable(Initializers.glorot(
            Shape(in_size=self.shape.in_size, out_size=self.shape.out_size, batch_size=self.shape.batch_size)),
            trainable=True, name="GraphConvolution:W")
        self.b = tf.Variable(Initializers.glorot(
            Shape(in_size=self.shape.batch_size, out_size=self.shape.out_size, batch_size=self.shape.batch_size)),
            trainable=True, name="GraphConvolution:b")

    def collect(self):
        return [self.W, self.b]

    def compute(self, parameters):
        self.output = tf.add(tf.matmul(tf.matmul(self.adjacency, self.input.output), parameters[0]), parameters[1])


class DirectionalGraphConvolution(Trainable):
    def __init__(self, input, in_adjacency, out_adjacency, shape: Shape):
        super(DirectionalGraphConvolution, self).__init__("DirectionalGraphConvolution", shape)
        self.input = input
        self.shape = shape
        self.output = tf.Variable(Initializers.glorot(self.shape), trainable=True)
        self.in_adjacency = in_adjacency
        self.out_adjacency = out_adjacency
        self.W_all = tf.Variable(Initializers.glorot(
            Shape(in_size=self.shape.in_size, out_size=self.shape.out_size, batch_size=self.shape.batch_size), axis=0),
            trainable=True, name="DirectionalGraphConvolution:W_all")
        self.b_all = tf.Variable(Initializers.glorot(
            Shape(in_size=self.shape.batch_size, out_size=self.shape.out_size, batch_size=self.shape.batch_size),
            axis=1), trainable=True, name="DirectionalGraphConvolution:b_all")
        self.W_in = tf.Variable(Initializers.glorot(
            Shape(in_size=self.shape.in_size, out_size=self.shape.out_size, batch_size=self.shape.batch_size), axis=0),
            trainable=True, name="DirectionalGraphConvolution:W_in")
        self.b_in = tf.Variable(Initializers.glorot(
            Shape(in_size=self.shape.batch_size, out_size=self.shape.out_size, batch_size=self.shape.batch_size),
            axis=1), trainable=True, name="DirectionalGraphConvolution:b_in")
        self.W_out = tf.Variable(Initializers.glorot(
            Shape(in_size=self.shape.in_size, out_size=self.shape.out_size, batch_size=self.shape.batch_size), axis=0),
            trainable=True, name="DirectionalGraphConvolution:W_out")
        self.b_out = tf.Variable(Initializers.glorot(
            Shape(in_size=self.shape.batch_size, out_size=self.shape.out_size, batch_size=self.shape.batch_size),
            axis=1), trainable=True, name="DirectionalGraphConvolution:b_out")
        self.C_in = tf.Variable(Initializers.glorot(Shape(in_size=1, out_size=1, batch_size=1)), trainable=True,
                                name="DirectionalGraphConvolution:C_in")
        self.C_out = tf.Variable(Initializers.glorot(Shape(in_size=1, out_size=1, batch_size=1)), trainable=True,
                                 name="DirectionalGraphConvolution:C_out")

    def collect(self):
        return [self.W_all, self.b_all, self.W_in, self.b_in, self.W_out, self.b_out, self.C_in, self.C_out]

    def compute(self, parameters):
        in_term = tf.add(tf.matmul(tf.matmul(self.in_adjacency, self.input.output), parameters[2]), parameters[3])
        out_term = tf.add(tf.matmul(tf.matmul(self.out_adjacency, self.input.output), parameters[4]), parameters[5])
        error_in_term = tf.add(tf.matmul(tf.matmul(self.in_adjacency, self.input.output), parameters[0]), parameters[1])
        error_out_term = tf.add(tf.matmul(tf.matmul(self.out_adjacency, self.input.output), parameters[0]),
                                parameters[1])
        in_term = tf.add(in_term, error_in_term)
        out_term = tf.add(out_term, error_out_term)
        self.output = tf.add(tf.multiply(parameters[6], in_term), tf.multiply(parameters[7], out_term))


class Model:
    def __init__(self):
        self.hp = {}
        self.epochs = {}
        self.predictions = {}
        self.valid_loss = {}
        self.valid_accuracy = {}
        self.embedding = {}
        self.parameters = {}
        self.losses = {}
        self.accuracies = {}

    def prepare(self):
        pass

    def compile(self):
        pass

    def compute(self):
        pass

    def collect(self):
        pass

    def evaluate(self):
        pass

    def update(self):
        pass

    def train(self):
        pass

    def validate(self):
        pass

    def visualize(self):
        tsne = skl.manifold.TSNE(n_components=2, perplexity=3, learning_rate=10)
        tsne.fit_transform(np.array(self.embedding))
        x = tsne.embedding_[:, 0]
        y = tsne.embedding_[:, 1]
        embedding_df = pd.DataFrame({"x": x, "y": y})
        losses_df = pd.DataFrame({"epochs": self.epochs, "losses": self.losses})
        accuracies_df = pd.DataFrame({"epochs": self.epochs, "accuracies": self.accuracies})
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        fig.suptitle('Model Visualization')
        sns.scatterplot(ax=axes[0], data=embedding_df, x='x', y='y')
        sns.lineplot(ax=axes[1], data=losses_df, x='epochs', y='losses')
        sns.lineplot(ax=axes[2], data=accuracies_df, x='epochs', y='accuracies')
        plt.show()

    def run(self):
        all_accuracies = []
        all_embeddings = []
        for _ in range(self.hp.get("trials")):
            valid_embeddings = []
            valid_losses = []
            valid_accuracies = []
            valid_predictions = []
            valid_parameters = []
            epochs = []
            losses = []
            accuracies = []
            for k in range(self.hp.get("K")):
                print(str(self.hp.get("K")) + " fold validation iteration number: " + str(k))
                self.prepare()
                self.compile()
                self.collect()
                self.train()
                self.validate()
                for sublist in self.predictions:
                    if np.isnan(sublist[0]):
                        print("The list contains a NaN")
                        break
                    else:
                        valid_predictions.append(self.predictions)
                        valid_losses.append(self.valid_loss)
                        valid_accuracies.append(self.valid_accuracy)
                        valid_embeddings.append(self.embedding)
                        valid_parameters.append(self.parameters)
                for e in self.epochs:
                    epochs.append(e)
                for l in self.losses:
                    losses.append(l)
                for a in self.accuracies:
                    accuracies.append(a)
            if len(valid_accuracies) != 0:
                max_valid_accuracy_index = np.argmax(np.array(valid_accuracies))
                final_predictions = list(valid_predictions[max_valid_accuracy_index].numpy())
                final_loss = valid_losses[max_valid_accuracy_index].numpy()
                final_accuracy = valid_accuracies[max_valid_accuracy_index].numpy()
                final_embedding = list(valid_embeddings[max_valid_accuracy_index].numpy())
                final_parameters = valid_parameters[max_valid_accuracy_index]
            else:
                final_predictions = [0]
                final_loss = 0
                final_accuracy = 0
                final_embedding = 0
                final_parameters = [0]
            print("Final Loss: " + str(final_loss))
            print("Final Accuracy: " + str(final_accuracy))
            all_accuracies.append(float(final_accuracy))
            all_embeddings.append(final_embedding)
            if final_embedding != 0:
                self.visualize()
        print("Average Accuracy: " + str(np.mean(np.array(all_accuracies))))
        return all_embeddings
