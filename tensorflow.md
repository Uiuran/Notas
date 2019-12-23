# TensorFlow Cheatsheet

####
## Install with GPU
####

Main-Ref
https://www.tensorflow.org/install/gpu

Important thing is your NVidia Driver be compatible version, that is a version greater than the CUDA driver, like in here i installed 440.39 for a CUDA 440.33.0 driver.
Since NVidia drivers for linux are a pain in the ass (for example, they can lead to behaviours such unidentifying your notebook screen), beaware to know the CUDA Toolkits driver version, you can find it in CUDA Driver section at https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html. 

See if your version of the driver is correct in relation to the listed in the reference above. My case Nvidia driver.
```bash 
nvidia-settings -v
```

Caveat on dependencies:
If you get stuck in installation process follow the link
https://devtalk.nvidia.com/default/topic/1043184/cuda-install-unmet-dependencies-cuda-depends-cuda-10-0-gt-10-0-130-but-it-is-not-going-to-be-installed/

That is, you must track the unmet/uninstalled dependency by doing ```bash sudo apt-get install xxxx``` until you find it. In my case adding universe repo and installing freeglut3 package was not enough, i had to track, always try to install the intended package after sucessfully installing a dependency, until you got it all installed.
You will eventually have to add repos not from your gpu vendor to install all the toolchain, for example:
```bash
sudo add-apt-repository universe
sudo apt-get update
sudo apt-get install freeglut3
```
Finally

```bash
pip install tensorflow-gpu
```

### Test Tensorflow use of your GPU

Execute the following python script 

```python
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
```
You should see something like that (if you have only one GPU NVidia GTX 1050):

```bash
2019-11-29 01:34:58.001393: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
2019-11-29 01:34:58.101668: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1006] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2019-11-29 01:34:58.102102: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties: 
name: GeForce GTX 1050 major: 6 minor: 1 memoryClockRate(GHz): 1.442
pciBusID: 0000:01:00.0
2019-11-29 01:34:58.102278: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
2019-11-29 01:34:58.103293: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
2019-11-29 01:34:58.104170: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10.0
2019-11-29 01:34:58.104373: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10.0
2019-11-29 01:34:58.105444: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10.0
2019-11-29 01:34:58.106288: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10.0
2019-11-29 01:34:58.218997: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2019-11-29 01:34:58.219293: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1006] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2019-11-29 01:34:58.220520: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:1006] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2019-11-29 01:34:58.221532: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
Num GPUs Available:  1
```

####
## Activate Tensorboard
####

```python
/some/path/mnist_experiments/
/some/path/mnist_experiments/run1/
/some/path/mnist_experiments/run1/events.out.tfevents.1456525581.name
/some/path/mnist_experiments/run1/events.out.tfevents.1456525585.name
/some/path/mnist_experiments/run2/
/some/path/mnist_experiments/run2/events.out.tfevents.1456525385.name
/tensorboard --logdir /some/path/mnist_experiments
```

```python
tensorboard --logdir name1:/path/to/logs/1,name2:/path/to/logs/2 --port 6006 --debugger_port 6064
``` 

- Insert [`tf.summary.scalar`](https://www.tensorflow.org/api_docs/python/tf/summary/scalar) nodes in the output nodes of the quantities you gonna supervise with tensorflow such


```python
from tensorflow.python import debug as tf_debug

session = tf_debug.TensorBoardDebugWrapperSession(session,"grpc://localhost:6064")

with tf.summary.FileWriter('/home/penalvad/stattus4/stattus4-audio-models/notebooks/',graph=buildgraph.graph,session=session) as writer:
...
```

acessar
localhost:6006

####
## CPP API for Operations DOCS:
####

https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/resource-scatter-nd-add

####
## Handy tensorflow functions to Comp Graph editing
####

tf.graph_util.extract_sub_graph(
    graph_def,
    dest_nodes
)
graph_def: A graph_pb2.GraphDef proto.
dest_nodes: A list of strings specifying the destination node names.
Returns:
The GraphDef of the sub-graph.

####
## Import Static Graph from GraphDef proto, i.e. graph without trainable variables
####

tf.graph_util.import_graph_def(
    graph_def,
    input_map=None,
    return_elements=None,
    name=None,
    op_dict=None,
    producer_op_list=None
)
graph_def: A GraphDef proto containing operations to be imported into the default graph.
input_map: A dictionary mapping input names (as strings) in graph_def to Tensor objects. The values of the named input tensors in the imported graph will be re-mapped to the respective Tensor values.
return_elements: A list of strings containing operation names in graph_def that will be returned as Operation objects; and/or tensor names in graph_def that will be returned as Tensor objects.
name: (Optional.) A prefix that will be prepended to the names in graph_def. Note that this does not apply to imported function names. Defaults to "import".
op_dict: (Optional.) Deprecated, do not use.
producer_op_list: (Optional.) An OpList proto with the (possibly stripped) list of OpDefs used by the producer of the graph. If provided, unrecognized attrs for ops in graph_def that have their default value according to producer_op_list will be removed. This will allow some more GraphDefs produced by later binaries to be accepted by earlier binaries.
Returns:
A list of Operation and/or Tensor objects from the imported graph, corresponding to the names in return_elements, and None if returns_elements is None.

####
## Variable Scope
####
Class VariableScope
Variable scope object to carry defaults to provide to get_variable.

Aliases:
Class tf.compat.v1.VariableScope
Many of the arguments we need for get_variable in a variable store are most easily handled with a context. This object is used for the defaults.

Attributes:
name: name of the current scope, used as prefix in get_variable.
initializer: default initializer passed to get_variable.
regularizer: default regularizer passed to get_variable.
reuse: Boolean, None, or tf.compat.v1.AUTO_REUSE, setting the reuse in get_variable. When eager execution is enabled this argument is always forced to be False.
caching_device: string, callable, or None: the caching device passed to get_variable.
partitioner: callable or None: the partitioner passed to get_variable.
custom_getter: default custom getter passed to get_variable.
name_scope: The name passed to tf.name_scope.
dtype: default type passed to get_variable (defaults to DT_FLOAT).
use_resource: if False, create a normal Variable; if True create an experimental ResourceVariable with well-defined semantics. Defaults to False (will later change to True). When eager execution is enabled this argument is always forced to be True.
constraint: An optional projection function to be applied to the variable after being updated by an Optimizer (e.g. used to implement norm constraints or value constraints for layer weights). The function must take as input the unprojected Tensor representing the value of the variable and return the Tensor for the projected value (which must have the same shape). Constraints are not safe to use when doing asynchronous distributed training.
__init__
View source

__init__(
    reuse,
    name='',
    initializer=None,
    regularizer=None,
    caching_device=None,
    partitioner=None,
    custom_getter=None,
    name_scope='',
    dtype=tf.dtypes.float32,
    use_resource=None,
    constraint=None
)

Creates a new VariableScope with the given properties.

Properties
caching_device
constraint
custom_getter
dtype
initializer
name
original_name_scope
partitioner
regularizer
reuse
use_resource
Methods
get_collection
View source

get_collection(name)

Get this scope's variables.

get_variable
View source

get_variable(
    var_store,
    name,
    shape=None,
    dtype=None,
    initializer=None,
    regularizer=None,
    reuse=None,
    trainable=None,
    collections=None,
    caching_device=None,
    partitioner=None,
    validate_shape=True,
    use_resource=None,
    custom_getter=None,
    constraint=None,
    synchronization=tf.VariableSynchronization.AUTO,
    aggregation=tf.VariableAggregation.NONE
)

Gets an existing variable with this name or create a new one.

global_variables
View source

global_variables()

Get this scope's global variables.

local_variables
View source

local_variables()

Get this scope's local variables.

reuse_variables
View source

reuse_variables()

Reuse variables in this scope.

set_caching_device
View source

set_caching_device(caching_device)

Set caching_device for this scope.

set_custom_getter
View source

set_custom_getter(custom_getter)

Set custom getter for this scope.

set_dtype
View source

set_dtype(dtype)

Set data type for this scope.

set_initializer
View source

set_initializer(initializer)

Set initializer for this scope.

set_partitioner
View source

set_partitioner(partitioner)

Set partitioner for this scope.

set_regularizer
View source

set_regularizer(regularizer)

Set regularizer for this scope.

set_use_resource
View source

set_use_resource(use_resource)

Sets whether to use ResourceVariables for this scope.

trainable_variables
View source

trainable_variables()

Get this scope's trainable variables.




####
## Add To Collections: add some collection such 'variables':[<tf.Variables ...>,<tf.Variables...>,....]
####

tf.add_to_collections(
    names,
    value
)

See tf.Graph.add_to_collections for more details.

Args:
names: The key for the collections. The GraphKeys class contains many standard names for collections.
value: The value to add to the collections.

####
## Saver, saver and restore variables in Meta Graph Def Proto files
####

See Variables for an overview of variables, saving and restoring.

The Saver class adds ops to save and restore variables to and from checkpoints. It also provides convenience methods to run these ops.

Checkpoints are binary files in a proprietary format which map variable names to tensor values. The best way to examine the contents of a checkpoint is to load it using a Saver.

Savers can automatically number checkpoint filenames with a provided counter. This lets you keep multiple checkpoints at different steps while training a model. For example you can number the checkpoint filenames with the training step number. To avoid filling up disks, savers manage checkpoint files automatically. For example, they can keep only the N most recent files, or one checkpoint for every N hours of training.

You number checkpoint filenames by passing a value to the optional global_step argument to save():

saver.save(sess, 'my-model', global_step=0) ==> filename: 'my-model-0'
...
saver.save(sess, 'my-model', global_step=1000) ==> filename: 'my-model-1000'

Additionally, optional arguments to the Saver() constructor let you control the proliferation of checkpoint files on disk:

max_to_keep indicates the maximum number of recent checkpoint files to keep. As new files are created, older files are deleted. If None or 0, no checkpoints are deleted from the filesystem but only the last one is kept in the checkpoint file. Defaults to 5 (that is, the 5 most recent checkpoint files are kept.)

keep_checkpoint_every_n_hours: In addition to keeping the most recent max_to_keep checkpoint files, you might want to keep one checkpoint file for every N hours of training. This can be useful if you want to later analyze how a model progressed during a long training session. For example, passing keep_checkpoint_every_n_hours=2 ensures that you keep one checkpoint file for every 2 hours of training. The default value of 10,000 hours effectively disables the feature.

Note that you still have to call the save() method to save the model. Passing these arguments to the constructor will not save variables automatically for you.

A training program that saves regularly looks like:

...
# Create a saver.
saver = tf.compat.v1.train.Saver(...variables...)
# Launch the graph and train, saving the model every 1,000 steps.
sess = tf.compat.v1.Session()
for step in xrange(1000000):
    sess.run(..training_op..)
    if step % 1000 == 0:
        # Append the step number to the checkpoint name:
        saver.save(sess, 'my-model', global_step=step)

In addition to checkpoint files, savers keep a protocol buffer on disk with the list of recent checkpoints. This is used to manage numbered checkpoint files and by latest_checkpoint(), which makes it easy to discover the path to the most recent checkpoint. That protocol buffer is stored in a file named 'checkpoint' next to the checkpoint files.

If you create several savers, you can specify a different filename for the protocol buffer file in the call to save().

__init__
View source

__init__(
    var_list=None,
    reshape=False,
    sharded=False,
    max_to_keep=5,
    keep_checkpoint_every_n_hours=10000.0,
    name=None,
    restore_sequentially=False,
    saver_def=None,
    builder=None,
    defer_build=False,
    allow_empty=False,
    write_version=tf.train.SaverDef.V2,
    pad_step_number=False,
    save_relative_paths=False,
    filename=None
)

Creates a Saver.

The constructor adds ops to save and restore variables.

var_list specifies the variables that will be saved and restored. It can be passed as a dict or a list:

A dict of names to variables: The keys are the names that will be used to save or restore the variables in the checkpoint files.
A list of variables: The variables will be keyed with their op name in the checkpoint files.
For example:
v1 = tf.Variable(..., name='v1')
v2 = tf.Variable(..., name='v2')

# Pass the variables as a dict:
saver = tf.compat.v1.train.Saver({'v1': v1, 'v2': v2})

# Or pass them as a list.
saver = tf.compat.v1.train.Saver([v1, v2])
# Passing a list is equivalent to passing a dict with the variable op names
# as keys:
saver = tf.compat.v1.train.Saver({v.op.name: v for v in [v1, v2]})

The optional reshape argument, if True, allows restoring a variable from a save file where the variable had a different shape, but the same number of elements and type. This is useful if you have reshaped a variable and want to reload it from an older checkpoint.

The optional sharded argument, if True, instructs the saver to shard checkpoints per device.

Args:
var_list: A list of Variable/SaveableObject, or a dictionary mapping names to SaveableObjects. If None, defaults to the list of all saveable objects.
reshape: If True, allows restoring parameters from a checkpoint where the variables have a different shape.
sharded: If True, shard the checkpoints, one per device.
max_to_keep: Maximum number of recent checkpoints to keep. Defaults to 5.
keep_checkpoint_every_n_hours: How often to keep checkpoints. Defaults to 10,000 hours.
name: String. Optional name to use as a prefix when adding operations.
restore_sequentially: A Bool, which if true, causes restore of different variables to happen sequentially within each device. This can lower memory usage when restoring very large models.
saver_def: Optional SaverDef proto to use instead of running the builder. This is only useful for specialty code that wants to recreate a Saver object for a previously built Graph that had a Saver. The saver_def proto should be the one returned by the as_saver_def() call of the Saver that was created for that Graph.
builder: Optional SaverBuilder to use if a saver_def was not provided. Defaults to BulkSaverBuilder().
defer_build: If True, defer adding the save and restore ops to the build() call. In that case build() should be called before finalizing the graph or using the saver.
allow_empty: If False (default) raise an error if there are no variables in the graph. Otherwise, construct the saver anyway and make it a no-op.
write_version: controls what format to use when saving checkpoints. It also affects certain filepath matching logic. The V2 format is the recommended choice: it is much more optimized than V1 in terms of memory required and latency incurred during restore. Regardless of this flag, the Saver is able to restore from both V2 and V1 checkpoints.
pad_step_number: if True, pads the global step number in the checkpoint filepaths to some fixed width (8 by default). This is turned off by default.
save_relative_paths: If True, will write relative paths to the checkpoint state file. This is needed if the user wants to copy the checkpoint directory and reload from the copied directory.
filename: If known at graph construction time, filename used for variable loading/saving.
Raises:
TypeError: If var_list is invalid.
ValueError: If any of the keys or values in var_list are not unique.
RuntimeError: If eager execution is enabled andvar_list does not specify a list of varialbes to save.
Eager Compatibility
When eager execution is enabled, var_list must specify a list or dict of variables to save. Otherwise, a RuntimeError will be raised.

Although Saver works in some cases when executing eagerly, it is fragile. Please switch to tf.train.Checkpoint or tf.keras.Model.save_weights, which perform a more robust object-based saving. These APIs will load checkpoints written by Saver.

Properties
last_checkpoints
List of not-yet-deleted checkpoint filenames.

You can pass any of the returned values to restore().

Returns:
A list of checkpoint filenames, sorted from oldest to newest.

Methods
as_saver_def
View source

as_saver_def()

Generates a SaverDef representation of this saver.

Returns:
A SaverDef proto.

build
View source

build()

export_meta_graph
View source

####
## Meta Graph Def
####

export_meta_graph(
    filename=None,
    collection_list=None,
    as_text=False,
    export_scope=None,
    clear_devices=False,
    clear_extraneous_savers=False,
    strip_default_attrs=False,
    save_debug_info=False
)

Writes MetaGraphDef to save_path/filename.

Args:
filename: Optional meta_graph filename including the path.
collection_list: List of string keys to collect.
as_text: If True, writes the meta_graph as an ASCII proto.
export_scope: Optional string. Name scope to remove.
clear_devices: Whether or not to clear the device field for an Operation or Tensor during export.
clear_extraneous_savers: Remove any Saver-related information from the graph (both Save/Restore ops and SaverDefs) that are not associated with this Saver.
strip_default_attrs: Boolean. If True, default-valued attributes will be removed from the NodeDefs. For a detailed guide, see Stripping Default-Valued Attributes.
save_debug_info: If True, save the GraphDebugInfo to a separate file, which in the same directory of filename and with _debug added before the file extension.
Returns:
A MetaGraphDef proto.

####
## Import Meta Graph
####

tf.train.import_meta_graph(
    meta_graph_or_file,
    clear_devices=False,
    import_scope=None,
    **kwargs
)

from_proto
View source

@staticmethod
from_proto(
    saver_def,
    import_scope=None
)

Returns a Saver object created from saver_def.

Args:
saver_def: a SaverDef protocol buffer.
import_scope: Optional string. Name scope to use.
Returns:
A Saver built from saver_def.

recover_last_checkpoints
View source

recover_last_checkpoints(checkpoint_paths)

Recovers the internal saver state after a crash.

This method is useful for recovering the "self._last_checkpoints" state.

Globs for the checkpoints pointed to by checkpoint_paths. If the files exist, use their mtime as the checkpoint timestamp.

Args:
checkpoint_paths: a list of checkpoint paths.
restore
View source

restore(
    sess,
    save_path
)

Restores previously saved variables.

This method runs the ops added by the constructor for restoring variables. It requires a session in which the graph was launched. The variables to restore do not have to have been initialized, as restoring is itself a way to initialize variables.

The save_path argument is typically a value previously returned from a save() call, or a call to latest_checkpoint().

Args:
sess: A Session to use to restore the parameters. None in eager mode.
save_path: Path where parameters were previously saved.
Raises:
ValueError: If save_path is None or not a valid checkpoint.
save
View source

save(
    sess,
    save_path,
    global_step=None,
    latest_filename=None,
    meta_graph_suffix='meta',
    write_meta_graph=True,
    write_state=True,
    strip_default_attrs=False,
    save_debug_info=False
)

Saves variables.

This method runs the ops added by the constructor for saving variables. It requires a session in which the graph was launched. The variables to save must also have been initialized.

The method returns the path prefix of the newly created checkpoint files. This string can be passed directly to a call to restore().

Args:
sess: A Session to use to save the variables.
save_path: String. Prefix of filenames created for the checkpoint.
global_step: If provided the global step number is appended to save_path to create the checkpoint filenames. The optional argument can be a Tensor, a Tensor name or an integer.
latest_filename: Optional name for the protocol buffer file that will contains the list of most recent checkpoints. That file, kept in the same directory as the checkpoint files, is automatically managed by the saver to keep track of recent checkpoints. Defaults to 'checkpoint'.
meta_graph_suffix: Suffix for MetaGraphDef file. Defaults to 'meta'.
write_meta_graph: Boolean indicating whether or not to write the meta graph file.
write_state: Boolean indicating whether or not to write the CheckpointStateProto.
strip_default_attrs: Boolean. If True, default-valued attributes will be removed from the NodeDefs. For a detailed guide, see Stripping Default-Valued Attributes.
save_debug_info: If True, save the GraphDebugInfo to a separate file, which in the same directory of save_path and with _debug added before the file extension. This is only enabled when write_meta_graph is True
Returns:
A string: path prefix used for the checkpoint files. If the saver is sharded, this string ends with: '-?????-of-nnnnn' where 'nnnnn' is the number of shards created. If the saver is empty, returns None.

Raises:
TypeError: If sess is not a Session.
ValueError: If latest_filename contains path components, or if it collides with save_path.
RuntimeError: If save and restore ops weren't built.
set_last_checkpoints
View source

set_last_checkpoints(last_checkpoints)

DEPRECATED: Use set_last_checkpoints_with_time.

Sets the list of old checkpoint filenames.

Args:
last_checkpoints: A list of checkpoint filenames.
Raises:
AssertionError: If last_checkpoints is not a list.
set_last_checkpoints_with_time
View source

set_last_checkpoints_with_time(last_checkpoints_with_time)

Sets the list of old checkpoint filenames and timestamps.

Args:
last_checkpoints_with_time: A list of tuples of checkpoint filenames and timestamps.
Raises:
AssertionError: If last_checkpoints_with_time is not a list.
to_proto
View source

to_proto(export_scope=None)

Converts this Saver to a SaverDef protocol buffer.

Args:
export_scope: Optional string. Name scope to remove.
Returns:
A SaverDef protocol buffer.

####
## TF Data (tf.data)
####

- A data source constructs a Dataset from data stored in memory or in one or more files.
- A data transformation constructs a dataset from one or more tf.data.Dataset objects.

### Basic mechanics

To create an input pipeline, you must start with a data source. For example, to construct a Dataset from data in memory, you can use tf.data.Dataset.from_tensors() or tf.data.Dataset.from_tensor_slices(). Alternatively, if your input data is stored in a file in the recommended TFRecord format, you can use tf.data.TFRecordDataset().

Once you have a Dataset object, you can transform it into a new Dataset by chaining method calls on the tf.data.Dataset object. For example, you can apply per-element transformations such as Dataset.map(), and multi-element transformations such as Dataset.batch(). See the documentation for tf.data.Dataset for a complete list of transformations.

The Dataset object is a Python iterable.

### Dataset structure
A dataset contains elements that each have the same (nested) structure and the individual components of the structure can be of any type representable by tf.TypeSpec, including Tensor, SparseTensor, RaggedTensor, TensorArray, or Dataset.

The Dataset.element_spec property allows you to inspect the type of each element component. The property returns a nested structure of tf.TypeSpec objects, matching the structure of the element, which may be a single component, a tuple of components, or a nested tuple of components. For example:

dataset1 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([4, 10]))

dataset1.element_spec

TensorSpec(shape=(10,), dtype=tf.float32, name=None)

dataset2 = tf.data.Dataset.from_tensor_slices(
   (tf.random.uniform([4]),
    tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)))

dataset2.element_spec

(TensorSpec(shape=(), dtype=tf.float32, name=None),
 TensorSpec(shape=(100,), dtype=tf.int32, name=None))

dataset3 = tf.data.Dataset.zip((dataset1, dataset2))

dataset3.element_spec

(TensorSpec(shape=(10,), dtype=tf.float32, name=None),
 (TensorSpec(shape=(), dtype=tf.float32, name=None),
  TensorSpec(shape=(100,), dtype=tf.int32, name=None)))

### Dataset containing a sparse tensor.
dataset4 = tf.data.Dataset.from_tensors(tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4]))

dataset4.element_spec

SparseTensorSpec(TensorShape([3, 4]), tf.int32)

### Use value_type to see the type of value represented by the element spec
dataset4.element_spec.value_type

tensorflow.python.framework.sparse_tensor.SparseTensor

The Dataset transformations support datasets of any structure. When using the Dataset.map(), and Dataset.filter() transformations, which apply a function to each element, the element structure determines the arguments of the function:

dataset1 = tf.data.Dataset.from_tensor_slices(
    tf.random.uniform([4, 10], minval=1, maxval=10, dtype=tf.int32))

dataset1

<TensorSliceDataset shapes: (10,), types: tf.int32>

for z in dataset1:
  print(z.numpy())

[9 1 7 2 7 2 5 8 8 8]
[9 9 3 2 4 9 9 4 3 6]
[3 9 2 9 3 1 8 7 2 8]
[7 5 2 8 2 2 1 3 3 8]

dataset2 = tf.data.Dataset.from_tensor_slices(
   (tf.random.uniform([4]),
    tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)))

dataset2

<TensorSliceDataset shapes: ((), (100,)), types: (tf.float32, tf.int32)>

dataset3 = tf.data.Dataset.zip((dataset1, dataset2))

dataset3

<ZipDataset shapes: ((10,), ((), (100,))), types: (tf.int32, (tf.float32, tf.int32))>

for a, (b,c) in dataset3:
  print('shapes: {a.shape}, {b.shape}, {c.shape}'.format(a=a, b=b, c=c))

shapes: (10,), (), (100,)
shapes: (10,), (), (100,)
shapes: (10,), (), (100,)
shapes: (10,), (), (100,)

Reading input data
Consuming NumPy arrays
See Loading NumPy arrays for more examples.

If all of your input data fits in memory, the simplest way to create a Dataset from them is to convert them to tf.Tensor objects and use Dataset.from_tensor_slices().

train, test = tf.keras.datasets.fashion_mnist.load_data()

Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz
32768/29515 [=================================] - 0s 0us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz
26427392/26421880 [==============================] - 0s 0us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz
8192/5148 [===============================================] - 0s 0us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz
4423680/4422102 [==============================] - 0s 0us/step

images, labels = train
images = images/255

dataset = tf.data.Dataset.from_tensor_slices((images, labels))
dataset

<TensorSliceDataset shapes: ((28, 28), ()), types: (tf.float64, tf.uint8)>

### Consuming Python Generators

The Dataset.from_generator constructor converts the python generator to a fully functional tf.data.Dataset.

The constructor takes a callable as input, not an iterator. This allows it to restart the generator when it reaches the end. It takes an optional args argument, which is passed as the callable's arguments.

The output_types argument is required because tf.data builds a tf.Graph internally, and graph edges require a tf.dtype.

ds_counter = tf.data.Dataset.from_generator(count, args=[25], output_types=tf.int32, output_shapes = (), )

for count_batch in ds_counter.repeat().batch(10).take(10):
  print(count_batch.numpy())

[0 1 2 3 4 5 6 7 8 9]
[10 11 12 13 14 15 16 17 18 19]
[20 21 22 23 24  0  1  2  3  4]
[ 5  6  7  8  9 10 11 12 13 14]
[15 16 17 18 19 20 21 22 23 24]
[0 1 2 3 4 5 6 7 8 9]
[10 11 12 13 14 15 16 17 18 19]
[20 21 22 23 24  0  1  2  3  4]
[ 5  6  7  8  9 10 11 12 13 14]
[15 16 17 18 19 20 21 22 23 24]

The output_shapes argument is not required but is highly recomended as many tensorflow operations do not support tensors with unknown rank. If the length of a particular axis is unknown or variable, set it as None in the output_shapes.

It's also important to note that the output_shapes and output_types follow the same nesting rules as other dataset methods.

Here is an example generator that demonstrates both aspects, it returns tuples of arrays, where the second array is a vector with unknown length.

def gen_series():
  i = 0
  while True:
    size = np.random.randint(0, 10)
    yield i, np.random.normal(size=(size,))
    i += 1

for i, series in gen_series():
  print(i, ":", str(series))
  if i > 5:
    break

0 : []
1 : [ 1.4608 -0.0935 -1.6648 -0.1007 -0.7357 -1.3826  1.5566 -1.1551]
2 : [-1.5221  0.9483  0.0677 -0.2791  1.5543  0.1088]
3 : [ 1.6014 -0.9247 -1.4246  2.0607 -1.1251 -2.2987 -0.9535]
4 : [-0.3621  1.0529  0.017  -0.4316 -0.2567  1.0389  0.2873 -1.1755]
5 : [ 0.9343 -0.4901 -0.2022  0.6525]
6 : [ 0.2267 -0.291   0.5348 -0.7489 -1.5477 -0.4768 -1.6164 -0.21   -0.0965]

The first output is an int32 the second is a float32.

The first item is a scalar, shape (), and the second is a vector of unknown length, shape (None,)

ds_series = tf.data.Dataset.from_generator(
    gen_series, 
    output_types=(tf.int32, tf.float32), 
    output_shapes=((), (None,)))

ds_series

<DatasetV1Adapter shapes: ((), (None,)), types: (tf.int32, tf.float32)>

Now it can be used like a regular tf.data.Dataset. Note that when batching a dataset with a variable shape, you need to use Dataset.padded_batch.

ds_series_batch = ds_series.shuffle(20).padded_batch(10, padded_shapes=([], [None]))

ids, sequence_batch = next(iter(ds_series_batch))
print(ids.numpy())
print()
print(sequence_batch.numpy())

[13 14 12 19 17  3  0  7  6 25]

[[ 0.0699  2.4317  0.      0.      0.      0.      0.      0.    ]
 [-1.0459 -0.5841  0.9426  0.7589  0.2838  0.3323 -2.3105  2.0333]
 [-0.7891  0.4218 -1.0847 -1.2004 -0.0554  0.      0.      0.    ]
 [ 0.4331  2.3756 -0.2903  1.0903 -0.0372  0.0772  0.      0.    ]
 [-1.0121  0.6696  2.1943  1.7748  0.8892  0.9743  0.9654  0.    ]
 [-1.7664  0.5746  0.      0.      0.      0.      0.      0.    ]
 [-0.6153  0.      0.      0.      0.      0.      0.      0.    ]
 [-0.2552  0.0501 -0.5079 -1.4735  0.4816 -0.7831  0.      0.    ]
 [ 0.4108 -0.0345 -1.8443 -1.008   0.056   0.6    -0.6672  0.0302]
 [ 0.      0.      0.      0.      0.      0.      0.      0.    ]]

For a more realistic example, try wrapping preprocessing.image.ImageDataGenerator as a tf.data.Dataset.

First download the data:

flowers = tf.keras.utils.get_file(
    'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True)

Downloading data from https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
228818944/228813984 [==============================] - 2s 0us/step

Create the image.ImageDataGenerator

img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=20)

images, labels = next(img_gen.flow_from_directory(flowers))

Found 3670 images belonging to 5 classes.

print(images.dtype, images.shape)
print(labels.dtype, labels.shape)

float32 (32, 256, 256, 3)
float32 (32, 5)

ds = tf.data.Dataset.from_generator(
    img_gen.flow_from_directory, args=[flowers], 
    output_types=(tf.float32, tf.float32), 
    output_shapes=([32,256,256,3], [32,5])
)

ds

<DatasetV1Adapter shapes: ((32, 256, 256, 3), (32, 5)), types: (tf.float32, tf.float32)>

Consuming TFRecord data
See Loading TFRecords for an end-to-end example.

The tf.data API supports a variety of file formats so that you can process large datasets that do not fit in memory. For example, the TFRecord file format is a simple record-oriented binary format that many TensorFlow applications use for training data. The tf.data.TFRecordDataset class enables you to stream over the contents of one or more TFRecord files as part of an input pipeline.

Here is an example using the test file from the French Street Name Signs (FSNS).

### Creates a dataset that reads all of the examples from two files.
fsns_test_file = tf.keras.utils.get_file("fsns.tfrec", "https://storage.googleapis.com/download.tensorflow.org/data/fsns-20160927/testdata/fsns-00000-of-00001")

Downloading data from https://storage.googleapis.com/download.tensorflow.org/data/fsns-20160927/testdata/fsns-00000-of-00001
7905280/7904079 [==============================] - 1s 0us/step

The filenames argument to the TFRecordDataset initializer can either be a string, a list of strings, or a tf.Tensor of strings. Therefore if you have two sets of files for training and validation purposes, you can create a factory method that produces the dataset, taking filenames as an input argument:

dataset = tf.data.TFRecordDataset(filenames = [fsns_test_file])
dataset

<TFRecordDatasetV2 shapes: (), types: tf.string>

Many TensorFlow projects use serialized tf.train.Example records in their TFRecord files. These need to be decoded before they can be inspected:

raw_example = next(iter(dataset))
parsed = tf.train.Example.FromString(raw_example.numpy())

parsed.features.feature['image/text']

bytes_list {
  value: "Rue Perreyon"
}

Consuming text data
See Loading Text for an end to end example.

Many datasets are distributed as one or more text files. The tf.data.TextLineDataset provides an easy way to extract lines from one or more text files. Given one or more filenames, a TextLineDataset will produce one string-valued element per line of those files.

directory_url = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
file_names = ['cowper.txt', 'derby.txt', 'butler.txt']

file_paths = [
    tf.keras.utils.get_file(file_name, directory_url + file_name)
    for file_name in file_names
]

Downloading data from https://storage.googleapis.com/download.tensorflow.org/data/illiad/cowper.txt
819200/815980 [==============================] - 0s 0us/step
Downloading data from https://storage.googleapis.com/download.tensorflow.org/data/illiad/derby.txt
811008/809730 [==============================] - 0s 0us/step
Downloading data from https://storage.googleapis.com/download.tensorflow.org/data/illiad/butler.txt
811008/807992 [==============================] - 0s 0us/step

dataset = tf.data.TextLineDataset(file_paths)

Here are the first few lines of the first file:

for line in dataset.take(5):
  print(line.numpy())

b"\xef\xbb\xbfAchilles sing, O Goddess! Peleus' son;"
b'His wrath pernicious, who ten thousand woes'
b"Caused to Achaia's host, sent many a soul"
b'Illustrious into Ades premature,'
b'And Heroes gave (so stood the will of Jove)'

To alternate lines between files use Dataset.interleave. This makes it easier to shuffle files together. Here are the first, second and third lines from each translation:

files_ds = tf.data.Dataset.from_tensor_slices(file_paths)
lines_ds = files_ds.interleave(tf.data.TextLineDataset, cycle_length=3)

for i, line in enumerate(lines_ds.take(9)):
  if i % 3 == 0:
    print()
  print(line.numpy())


b"\xef\xbb\xbfAchilles sing, O Goddess! Peleus' son;"
b"\xef\xbb\xbfOf Peleus' son, Achilles, sing, O Muse,"
b'\xef\xbb\xbfSing, O goddess, the anger of Achilles son of Peleus, that brought'

b'His wrath pernicious, who ten thousand woes'
b'The vengeance, deep and deadly; whence to Greece'
b'countless ills upon the Achaeans. Many a brave soul did it send'

b"Caused to Achaia's host, sent many a soul"
b'Unnumbered ills arose; which many a soul'
b'hurrying down to Hades, and many a hero did it yield a prey to dogs and'

By default, a TextLineDataset yields every line of each file, which may not be desirable, for example, if the file starts with a header line, or contains comments. These lines can be removed using the Dataset.skip() or Dataset.filter() transformations. Here we skip the first line, then filter to find only survivors.

titanic_file = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")
titanic_lines = tf.data.TextLineDataset(titanic_file)

Downloading data from https://storage.googleapis.com/tf-datasets/titanic/train.csv
32768/30874 [===============================] - 0s 0us/step

for line in titanic_lines.take(10):
  print(line.numpy())

b'survived,sex,age,n_siblings_spouses,parch,fare,class,deck,embark_town,alone'
b'0,male,22.0,1,0,7.25,Third,unknown,Southampton,n'
b'1,female,38.0,1,0,71.2833,First,C,Cherbourg,n'
b'1,female,26.0,0,0,7.925,Third,unknown,Southampton,y'
b'1,female,35.0,1,0,53.1,First,C,Southampton,n'
b'0,male,28.0,0,0,8.4583,Third,unknown,Queenstown,y'
b'0,male,2.0,3,1,21.075,Third,unknown,Southampton,n'
b'1,female,27.0,0,2,11.1333,Third,unknown,Southampton,n'
b'1,female,14.0,1,0,30.0708,Second,unknown,Cherbourg,n'
b'1,female,4.0,1,1,16.7,Third,G,Southampton,n'

def survived(line):
  return tf.not_equal(tf.strings.substr(line, 0, 1), "0")

survivors = titanic_lines.skip(1).filter(survived)

for line in survivors.take(10):
  print(line.numpy())

b'1,female,38.0,1,0,71.2833,First,C,Cherbourg,n'
b'1,female,26.0,0,0,7.925,Third,unknown,Southampton,y'
b'1,female,35.0,1,0,53.1,First,C,Southampton,n'
b'1,female,27.0,0,2,11.1333,Third,unknown,Southampton,n'
b'1,female,14.0,1,0,30.0708,Second,unknown,Cherbourg,n'
b'1,female,4.0,1,1,16.7,Third,G,Southampton,n'
b'1,male,28.0,0,0,13.0,Second,unknown,Southampton,y'
b'1,female,28.0,0,0,7.225,Third,unknown,Cherbourg,y'
b'1,male,28.0,0,0,35.5,First,A,Southampton,y'
b'1,female,38.0,1,5,31.3875,Third,unknown,Southampton,n'

Consuming CSV data
See Loading CSV Files, and Loading Pandas DataFrames for more examples.

The CSV file format is a popular format for storing tabular data in plain text.

For example:

titanic_file = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")

df = pd.read_csv(titanic_file, index_col=None)
df.head()


If your data fits in memory the same Dataset.from_tensor_slices method works on dictionaries, allowing this data to be easily imported:

titanic_slices = tf.data.Dataset.from_tensor_slices(dict(df))

for feature_batch in titanic_slices.take(1):
  for key, value in feature_batch.items():
    print("  {!r:20s}: {}".format(key, value))

  'survived'          : 0
  'sex'               : b'male'
  'age'               : 22.0
  'n_siblings_spouses': 1
  'parch'             : 0
  'fare'              : 7.25
  'class'             : b'Third'
  'deck'              : b'unknown'
  'embark_town'       : b'Southampton'
  'alone'             : b'n'

A more scalable approach is to load from disk as necessary.

The tf.data module provides methods to extract records from one or more CSV files that comply with RFC 4180.

The experimental.make_csv_dataset function is the high level interface for reading sets of csv files. It supports column type inference and many other features, like batching and shuffling, to make usage simple.

titanic_batches = tf.data.experimental.make_csv_dataset(
    titanic_file, batch_size=4,
    label_name="survived")

WARNING:tensorflow:From /tmpfs/src/tf_docs_env/lib/python3.6/site-packages/tensorflow_core/python/data/experimental/ops/readers.py:521: parallel_interleave (from tensorflow.python.data.experimental.ops.interleave_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.data.Dataset.interleave(map_func, cycle_length, block_length, num_parallel_calls=tf.data.experimental.AUTOTUNE)` instead. If sloppy execution is desired, use `tf.data.Options.experimental_determinstic`.
WARNING:tensorflow:From /tmpfs/src/tf_docs_env/lib/python3.6/site-packages/tensorflow_core/python/data/experimental/ops/readers.py:215: shuffle_and_repeat (from tensorflow.python.data.experimental.ops.shuffle_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.data.Dataset.shuffle(buffer_size, seed)` followed by `tf.data.Dataset.repeat(count)`. Static tf.data optimizations will take care of using the fused implementation.

for feature_batch, label_batch in titanic_batches.take(1):
  print("'survived': {}".format(label_batch))
  print("features:")
  for key, value in feature_batch.items():
    print("  {!r:20s}: {}".format(key, value))

'survived': [0 1 0 0]
features:
  'sex'               : [b'male' b'male' b'male' b'male']
  'age'               : [28. 20. 28. 40.]
  'n_siblings_spouses': [0 0 0 0]
  'parch'             : [0 0 0 0]
  'fare'              : [ 8.4583  7.2292  8.6625 27.7208]
  'class'             : [b'Third' b'Third' b'Third' b'First']
  'deck'              : [b'unknown' b'unknown' b'unknown' b'unknown']
  'embark_town'       : [b'Queenstown' b'Cherbourg' b'Southampton' b'Cherbourg']
  'alone'             : [b'y' b'y' b'y' b'y']

You can use the select_columns argument if you only need a subset of columns.

titanic_batches = tf.data.experimental.make_csv_dataset(
    titanic_file, batch_size=4,
    label_name="survived", select_columns=['class', 'fare', 'survived'])

for feature_batch, label_batch in titanic_batches.take(1):
  print("'survived': {}".format(label_batch))
  for key, value in feature_batch.items():
    print("  {!r:20s}: {}".format(key, value))

'survived': [0 1 1 1]
  'fare'              : [ 7.8542 23.25   14.5    20.575 ]
  'class'             : [b'Third' b'Third' b'Second' b'Third']

There is also a lower-level experimental.CsvDataset class which provides finer grained control. It does not support column type inference. Instead you must specify the type of each column.

titanic_types  = [tf.int32, tf.string, tf.float32, tf.int32, tf.int32, tf.float32, tf.string, tf.string, tf.string, tf.string] 
dataset = tf.data.experimental.CsvDataset(titanic_file, titanic_types , header=True)

for line in dataset.take(10):
  print([item.numpy() for item in line])

[0, b'male', 22.0, 1, 0, 7.25, b'Third', b'unknown', b'Southampton', b'n']
[1, b'female', 38.0, 1, 0, 71.2833, b'First', b'C', b'Cherbourg', b'n']
[1, b'female', 26.0, 0, 0, 7.925, b'Third', b'unknown', b'Southampton', b'y']
[1, b'female', 35.0, 1, 0, 53.1, b'First', b'C', b'Southampton', b'n']
[0, b'male', 28.0, 0, 0, 8.4583, b'Third', b'unknown', b'Queenstown', b'y']
[0, b'male', 2.0, 3, 1, 21.075, b'Third', b'unknown', b'Southampton', b'n']
[1, b'female', 27.0, 0, 2, 11.1333, b'Third', b'unknown', b'Southampton', b'n']
[1, b'female', 14.0, 1, 0, 30.0708, b'Second', b'unknown', b'Cherbourg', b'n']
[1, b'female', 4.0, 1, 1, 16.7, b'Third', b'G', b'Southampton', b'n']
[0, b'male', 20.0, 0, 0, 8.05, b'Third', b'unknown', b'Southampton', b'y']

If some columns are empty, this low-level interface allows you to provide default values instead of column types.

%%writefile missing.csv
1,2,3,4
,2,3,4
1,,3,4
1,2,,4
1,2,3,
,,,

Writing missing.csv

### Creates a dataset that reads all of the records from two CSV files, each with four float columns which may have missing values.

record_defaults = [999,999,999,999]
dataset = tf.data.experimental.CsvDataset("missing.csv", record_defaults)
dataset = dataset.map(lambda *items: tf.stack(items))
dataset

<MapDataset shapes: (4,), types: tf.int32>

for line in dataset:
  print(line.numpy())

[1 2 3 4]
[999   2   3   4]
[  1 999   3   4]
[  1   2 999   4]
[  1   2   3 999]
[999 999 999 999]

By default, a CsvDataset yields every column of every line of the file, which may not be desirable, for example if the file starts with a header line that should be ignored, or if some columns are not required in the input. These lines and fields can be removed with the header and select_cols arguments respectively.

# Creates a dataset that reads all of the records from two CSV files with
# headers, extracting float data from columns 2 and 4.
record_defaults = [999, 999] # Only provide defaults for the selected columns
dataset = tf.data.experimental.CsvDataset("missing.csv", record_defaults, select_cols=[1, 3])
dataset = dataset.map(lambda *items: tf.stack(items))
dataset

<MapDataset shapes: (2,), types: tf.int32>

for line in dataset:
  print(line.numpy())

[2 4]
[2 4]
[999   4]
[2 4]
[  2 999]
[999 999]

Consuming sets of files
There are many datasets distributed as a set of files, where each file is an example.

flowers_root = tf.keras.utils.get_file(
    'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True)
flowers_root = pathlib.Path(flowers_root)

Note: these images are licensed CC-BY, see LICENSE.txt for details.
The root directory contains a directory for each class:

for item in flowers_root.glob("*"):
  print(item.name)

daisy
LICENSE.txt
roses
dandelion
sunflowers
tulips

The files in each class directory are examples:

list_ds = tf.data.Dataset.list_files(str(flowers_root/'*/*'))

for f in list_ds.take(5):
  print(f.numpy())

b'/home/kbuilder/.keras/datasets/flower_photos/dandelion/14761980161_2d6dbaa4bb_m.jpg'
b'/home/kbuilder/.keras/datasets/flower_photos/roses/6783408274_974796e92f.jpg'
b'/home/kbuilder/.keras/datasets/flower_photos/dandelion/13897156242_dca5d93075_m.jpg'
b'/home/kbuilder/.keras/datasets/flower_photos/daisy/17101762155_2577a28395.jpg'
b'/home/kbuilder/.keras/datasets/flower_photos/sunflowers/9783416751_b2a03920f7_n.jpg'

We can read the data using the tf.io.read_file function and extract the label from the path, returning (image, label) pairs:

def process_path(file_path):
  label = tf.strings.split(file_path, '/')[-2]
  return tf.io.read_file(file_path), label

labeled_ds = list_ds.map(process_path)

for image_raw, label_text in labeled_ds.take(1):
  print(repr(image_raw.numpy()[:100]))
  print()
  print(label_text.numpy())

b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xfe\x00\x1ccmp3.10.3.1Lq3 0x1f7d625b\x00\xff\xdb\x00C\x00\x03\x02\x05\x03\x05\x02\x03\x03\x03\x03\x04\x07\x03\x04\x05\x08\x05\x05\t\t\x05\x0b\x07\x0f\x06\x08\r\x0b\r\r\x17\x0b\x0c\x0c\x0e\x1d%\x11\x0e\x0f#\x0f\x0c\x0c\x11'

b'tulips'

Batching dataset elements
Simple batching
The simplest form of batching stacks n consecutive elements of a dataset into a single element. The Dataset.batch() transformation does exactly this, with the same constraints as the tf.stack() operator, applied to each component of the elements: i.e. for each component i, all elements must have a tensor of the exact same shape.

inc_dataset = tf.data.Dataset.range(100)
dec_dataset = tf.data.Dataset.range(0, -100, -1)
dataset = tf.data.Dataset.zip((inc_dataset, dec_dataset))
batched_dataset = dataset.batch(4)

for batch in batched_dataset.take(4):
  print([arr.numpy() for arr in batch])

[array([0, 1, 2, 3]), array([ 0, -1, -2, -3])]
[array([4, 5, 6, 7]), array([-4, -5, -6, -7])]
[array([ 8,  9, 10, 11]), array([ -8,  -9, -10, -11])]
[array([12, 13, 14, 15]), array([-12, -13, -14, -15])]

While tf.data tries to propagate shape information, the default settings of Dataset.batch result in an unknown batch size because the last batch may not be full. Note the Nones in the shape:

batched_dataset

<BatchDataset shapes: ((None,), (None,)), types: (tf.int64, tf.int64)>

Use the drop_remainder argument to ignore that last batch, and get full shape propagation:

batched_dataset = dataset.batch(7, drop_remainder=True)
batched_dataset

<BatchDataset shapes: ((7,), (7,)), types: (tf.int64, tf.int64)>

Batching tensors with padding
The above recipe works for tensors that all have the same size. However, many models (e.g. sequence models) work with input data that can have varying size (e.g. sequences of different lengths). To handle this case, the Dataset.padded_batch() transformation enables you to batch tensors of different shape by specifying one or more dimensions in which they may be padded.

dataset = tf.data.Dataset.range(100)
dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
dataset = dataset.padded_batch(4, padded_shapes=(None,))

for batch in dataset.take(2):
  print(batch.numpy())
  print()

[[0 0 0]
 [1 0 0]
 [2 2 0]
 [3 3 3]]

[[4 4 4 4 0 0 0]
 [5 5 5 5 5 0 0]
 [6 6 6 6 6 6 0]
 [7 7 7 7 7 7 7]]


The Dataset.padded_batch() transformation allows you to set different padding for each dimension of each component, and it may be variable-length (signified by None in the example above) or constant-length. It is also possible to override the padding value, which defaults to 0.

### Colab Notebook

https://colab.research.google.com/drive/1Yo5kvh6OVV9KRTUEN9RFjCLX4EapWS-H
