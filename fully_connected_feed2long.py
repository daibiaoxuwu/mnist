import os
import time
import numpy as np

import tensorflow as tf

from mnistreader import reader
import mnist

FLAGS = None


batch_size=50
def placeholder_inputs(batch_size):
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 784))
  labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
  return images_placeholder, labels_placeholder


def fill_feed_dict(data_set, images_pl, labels_pl):
  """Fills the feed_dict for training the given step.

  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }

  Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().

  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
  # Create the feed_dict for the placeholders filled with the next
  # `batch size` examples.
  images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,
                                                 FLAGS.fake_data)
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
  }
  return feed_dict


def do_eval(sess, eval_correct,data_set,batch_size,images_placeholder,labels_placeholder,keep_prob):
  """Runs one evaluation against the full epoch of data.

  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
  """
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = data_set.readlength // FLAGS.batch_size 
  oldpointer= data_set.pointer
  data_set.pointer=data_set.readlength
  print(data_set.pointer)
  #steps_per_epoch = data_set.readlength // FLAGS.batch_size 

  num_examples = steps_per_epoch * FLAGS.batch_size
  for step in xrange(steps_per_epoch):
    inputs,answers=data_set.list_tags(batch_size,test=False)
    feed_dict= {
                images_placeholder:inputs,
                labels_placeholder:answers,
                keep_prob:1
                }

    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = float(true_count) / num_examples
  print('fakeeval Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))
  data_set.pointer=oldpointer




def do_evalfake(sess, eval_correct,data_set,batch_size,images_placeholder,labels_placeholder,logits,keep_prob):
  """Runs one evaluation against the full epoch of data.

  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
  """
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = data_set.readlength // FLAGS.batch_size // 6
  oldpointer= data_set.pointer
  data_set.pointer=data_set.readlength *5 //6
  
  #steps_per_epoch = data_set.readlength // FLAGS.batch_size 

  num_examples = steps_per_epoch * FLAGS.batch_size
  for step in xrange(steps_per_epoch):
  #  print('pointer1:',data_set.pointer)
    inputs,answers=data_set.list_tags(batch_size,test=True)
    feed_dict= {
                images_placeholder:inputs,
                labels_placeholder:answers,
                keep_prob:0.5
                }

    newcount,logi=sess.run([eval_correct,logits], feed_dict=feed_dict)
    true_count += newcount
    for i0 in range(FLAGS.batch_size):
            lgans=np.argmax(logi[i0])
            for i0 in range(FLAGS.batch_size):
                lgans=np.argmax(logi[i0])
                if(lgans!=answers[i0] and False):
                      for tt in range(784):
                          if(tt%28==0): print(' ');
                          if(inputs[i0][tt]!=0):
                              print('1',end=' ');
                          else:
                              print('0',end=' ');
#                      print('np',np.argmax(i),answers,answers[i0],'np')
                      print(lgans,answers[i0])
            # Update the events file.
  precision = float(true_count) / num_examples
  print('Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision),end='')
  data_set.pointer=oldpointer
  #print('pointer2:',data_set.pointer)



def run_training():
  """Train MNIST for a number of steps."""
  # Get the sets of images and labels for training, validation, and
  # test on MNIST.
  data_sets=reader(patchlength=0,\
            maxlength=300,\
            embedding_size=100,\
            num_verbs=10,\
            allinclude=False,\
            shorten=False,\
            shorten_front=False,\
            testflag=False,\
            passnum=0,\
            dpflag=False)

  
  

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Generate placeholders for the images and labels.
    images_placeholder, labels_placeholder = placeholder_inputs(
        FLAGS.batch_size)

    # Build a Graph that computes predictions from the inference model.
    logits,keep_prob = mnist.inference(images_placeholder,
                             FLAGS.hidden1,
                             FLAGS.hidden2)

    # Add to the Graph the Ops for loss calculation.
    loss = mnist.loss(logits, labels_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = mnist.training(loss, FLAGS.learning_rate)

    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = mnist.evaluation(logits, labels_placeholder)

    # Build the summary Tensor based on the TF collection of Summaries.
    summary = tf.summary.merge_all()

    # Add the variable initializer Op.
    init = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

    # And then after everything is built:

    # Run the Op to initialize the variables.
    with tf.Session() as session:
        sess.run(init)
        if True:
            model_file=tf.train.latest_checkpoint(FLAGS.log_dir)
            saver.restore(sess,model_file)

        # Start the training loop.
        start_time = time.time()
        for step in xrange(FLAGS.max_steps):

          # Fill a feed dictionary with the actual set of images and labels
          # for this particular training step.

        
          inputs,answers=data_sets.list_tags(FLAGS.batch_size,test=False)
#      print(len(inputs),len(inputs[0]),inputs[0])
#      input()
          inputs2=[]
          for i  in range(len(inputs)):
              inputs2.append(inputs[i]/255)
#      print(len(inputs2),len(inputs2[0]),inputs2[0])
#      input()
          feed_dict = {
              images_placeholder: inputs2,
              labels_placeholder: answers,
              keep_prob:0.5
          }
          # Run one step of the model.  The return values are the activations
          # from the `train_op` (which is discarded) and the `loss` Op.  To
          # inspect the values of your Ops or variables, you may include them
          # in the list passed to sess.run() and the value tensors will be
          # returned in the tuple from the call.
          _, loss_value,logi = sess.run([train_op, loss,logits],
                                   feed_dict=feed_dict)

          duration = time.time() - start_time

          # Write the summaries and print an overview fairly often.
          if step % 100 == 0:
            # Print status to stdout.
            print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
#            print(logi)
#            print(answers)
            for i0 in range(FLAGS.batch_size):
                lgans=np.argmax(logi[i0])
                if(lgans!=answers[i0] and False):
                      for tt in range(784):
                          if(tt%28==0): print(' ');
                          if(inputs[i0][tt]!=0):
                              print('1',end=' ');
                          else:
                              print('0',end=' ');
#                      print('np',np.argmax(i),answers,answers[i0],'np')
                      print(lgans,answers[i0])
            # Update the events file.
            summary_str = sess.run(summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()
          if (step + 1) % 500 == 0 or (step + 1) == FLAGS.max_steps:
            #print('Training Data Eval:')
            do_eval(sess,
                    eval_correct,data_sets,FLAGS.batch_size,
                    images_placeholder,
                    labels_placeholder,keep_prob)
            do_evalfake(sess,
                    eval_correct,data_sets,FLAGS.batch_size,
                    images_placeholder,
                    labels_placeholder,
                    logits,keep_prob)
          # Save a checkpoint and evaluate the model periodically.
          #if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
            checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
            saver.save(sess, checkpoint_file, global_step=step)
            print('saved to',checkpoint_file)
          '''
            # Evaluate against the training set.
            print('Training Data Eval:')
            do_eval(sess,
                    eval_correct,
                    images_placeholder,
                    labels_placeholder,
                    data_sets.train)
            # Evaluate against the validation set.
            print('Validation Data Eval:')
            do_eval(sess,
                    eval_correct,
                    images_placeholder,
                    labels_placeholder,
                    data_sets.validation)
            # Evaluate against the test set.
            print('Test Data Eval:')
            do_eval(sess,
                    eval_correct,
                    images_placeholder,
                    labels_placeholder,
                    data_sets.test)
            '''

def main(_):
#  if tf.gfile.Exists(FLAGS.log_dir):
#    tf.gfile.DeleteRecursively(FLAGS.log_dir)
#  tf.gfile.MakeDirs(FLAGS.log_dir)
  run_training()


if __name__ == '__main__':
  tf.app.run()
