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
