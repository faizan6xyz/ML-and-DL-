import tensorflow as tf

# =========================
# PART 1: WRITE TFRECORD
# =========================

# Example data
name = "Faizan"
age = 20

# Create a feature dictionary
# bytes_list is used for strings
# int64_list is used for integers
feature = {
    "name": tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[name.encode("utf-8")])
    ),
    "age": tf.train.Feature(
        int64_list=tf.train.Int64List(value=[age])
    )
}

# Create a TensorFlow Example object
example = tf.train.Example(
    features=tf.train.Features(feature=feature)
)

# Serialize the example into a binary string
serialized_example = example.SerializeToString()

# Write the serialized example to a TFRecord file
with tf.io.TFRecordWriter("data.tfrecord") as writer:
    writer.write(serialized_example)

print("TFRecord file created successfully.")

# =========================
# PART 2: READ TFRECORD
# =========================

# Define how the data is stored inside the TFRecord file
feature_description = {
    "name": tf.io.FixedLenFeature([], tf.string),
    "age": tf.io.FixedLenFeature([], tf.int64),
}

# Function to parse each record
def parse_function(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)

# Load the TFRecord file
dataset = tf.data.TFRecordDataset("data.tfrecord")

# Parse the records
dataset = dataset.map(parse_function)

# Read and print the data
for record in dataset:
    print("Name:", record["name"].numpy().decode("utf-8"))
    print("Age:", record["age"].numpy())