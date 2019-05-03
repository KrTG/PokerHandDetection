from modules.utility import SlidingBuffer

import tensorflow as tf
import numpy as np

class EarlyStoppingHook(tf.train.SessionRunHook):
    """
    Klasa monitorująca proces uczenia i zatrzymująca jeżeli błąd
    lub inny parametr nie poprawił się od dłuższego czasu
    """

    def __init__(self, monitor='val_loss', min_delta=0, patience=0, average=1,
                 mode='auto'):
        """
        Parametry
        ---------
        monitor: str, nazwa obserwowanego parametru
        min_delta: float, minimalna wartość o jaką musi zmienić się obserwowany
                parametr, by stwierdzić jego poprawę,
        patience: Ile kroków uczenia bez poprawy musi nastąpić do zatrzymania uczenia
        average: Ile kolejnych wartości parametru powinno zostać uśrednione
        mode: Jaka wartość obserwowanego parametru jest pożądana 
            ("min" - minimalna, "max" - maksymalna, "auto" - określone automatycznie na podstawie nazwy)
        """
        self.monitor = monitor
        self.patience = patience
        self.step = 0
        self.min_delta = min_delta
        self.wait = 0
        if mode not in ['auto', 'min', 'max']:
            print('EarlyStopping mode %s is unknown, '
                            'fallback to auto mode.', mode, RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.moving_window = SlidingBuffer.from_iterator(np.Inf if self.monitor_op == np.less else -np.Inf for _ in range(average))

    def begin(self):
        """
        Zapisuje referencję do monitorowanego parametru wyciągniętego z grafu obliczeniowego
        tensorflow. Wywoływana przed rozpoczęciem uczenia.        
        """
        # Convert names to tensors if given
        graph = tf.get_default_graph()
        self.monitor = graph.as_graph_element(self.monitor)
        if isinstance(self.monitor, tf.Operation):
            self.monitor = self.monitor.outputs[0]

    def before_run(self, run_context):  # pylint: disable=unused-argument
        """
        Metoda wywoływana przed pojedynczym krokiem uczenia.

        Parametry
        ---------
        run_context: nie używany
        """
        return tf.train.SessionRunArgs(self.monitor)

    def after_run(self, run_context, run_values):
        """
        Zapisuje wartość obserwowanego parametru do bufora,
        oblicza średnią wartości w buforze i jeżeli ta średnia
        nie spadła przez dłuższy czas, przerywa uczenie.
        Metoda wywoływana po pojedynczym kroku uczenia.

        Parametry
        ---------
        run_context: tensorflow.train.SessionRunContext, pozwala zatrzymać proces uczenia
        run_values: tensorflow.train.SessionRunValues, pozwala pobrać nową wartość obserwowanego parametru
        """        
        current = run_values.results
        self.moving_window.append(current)
        average = sum(self.moving_window) / len(self.moving_window)

        if self.monitor_op(average - self.min_delta, self.best):
            self.best = average
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print("Stopped due to early stopping hook!")
                run_context.request_stop()        
        self.step += 1
        if self.step % 400 == 0:
            print("Early stopping hook report: Min loss = {}".format(self.best))

class LearningRateFinderHook(tf.train.SessionRunHook):
    def __init__(self, loss_variable_name, num_steps, starting_rate=10**-8, final_rate=10, graph_smoothing=0.9):        
        self.loss_variable_name = loss_variable_name
        self.learning_rate = starting_rate
        self.beta = graph_smoothing        
        self.max_steps = num_steps
        self.multiplier = (final_rate / starting_rate)**(1 / num_steps)
        self.measurements = []
        self.avg_loss = None
        self.min_loss = 10**6

        self.current_step = 0

    def get_measurements(self):
        return self.measurements

    def after_create_session(self, session, coord):
        pass

    def before_run(self, run_context):
        run_context.session.run(self.assign_op, feed_dict={self.lrate_placeholder: self.learning_rate})      
        return tf.train.SessionRunArgs({'lrate': self.lrate, 'loss': self.loss_op.outputs[0]})

    def after_run(self, run_context, run_values):
        current = run_values.results        
        self.learning_rate *= self.multiplier

        if self.avg_loss is None:
            self.avg_loss = current['loss']
        else:
            self.avg_loss = self.beta * self.avg_loss + (1 - self.beta) * current['loss']
        
        if self.avg_loss < self.min_loss:
            self.min_loss = self.avg_loss
        if self.avg_loss > 10 * self.min_loss:
            run_context.request_stop()
        self.measurements.append((current['lrate'], self.avg_loss))           

        self.current_step += 1
        if self.current_step == self.max_steps:
            run_context.request_stop()     
        
    def begin(self):
        graph = tf.get_default_graph()

        self.loss_op = graph.as_graph_element(self.loss_variable_name)
        self.lrate = graph.get_tensor_by_name("lrate:0")        
        self.lrate_placeholder = tf.placeholder(tf.float32, [])
        self.assign_op = tf.assign(self.lrate, self.lrate_placeholder)
    
    def end(self, session):
        pass

class LearningSchedulerHook(tf.train.SessionRunHook):
    def __init__(self, batch_size, num_steps, min_rate, max_rate, crush_rate=10**-6, schedule=None, should_stop=False, resumed=True):
        if schedule is None:
            schedule = (0.45, 0.9, 1.0)
        else:
            if (len(schedule) != 3
            or not all(isinstance(x, (float, int)) for x in schedule)
            or not all(0 <= x <= 1 for x in schedule)
            or not (schedule[0] < schedule[1] and schedule[1] < schedule[2])):
                raise ValueError("Schedule must be a collection of 3 numbers in range [0,1], increasing")
        self.schedule = schedule
        self.should_stop = should_stop

        self.min_rate = min_rate
        self.max_rate = max_rate
        self.crush_rate = crush_rate

        self.num_steps = num_steps
        self.increment = (max_rate - min_rate) / (self.num_steps * self.schedule[0])
        self.decrement = (max_rate - min_rate) / (self.num_steps * (self.schedule[1] - self.schedule[0]))
        self.decrement_final = (min_rate - crush_rate) / (self.num_steps * (self.schedule[2] - self.schedule[1]))
        self.learning_rate = self.min_rate

        self.step = 0
        self.resumed = resumed
        

    def after_create_session(self, session, coord):        
        if self.resumed:
            self.step = session.run(tf.train.get_global_step())

            if self.step < self.num_steps * self.schedule[0]:
                self.learning_rate = self.min_rate + self.increment * self.step
            elif self.step < self.num_steps * self.schedule[1]:
                self.learning_rate = self.max_rate - self.decrement * (self.step - self.num_steps * self.schedule[0])
            elif self.step < self.num_steps * self.schedule[2]:
                self.learning_rate = self.min_rate - self.decrement_final * (self.step - self.num_steps * self.schedule[1])
            else:
                self.learning_rate = self.crush_rate


    def before_run(self, run_context):  # pylint: disable=unused-argument      
        run_context.session.run(self.assign_op, feed_dict={self.lrate_placeholder: self.learning_rate})      
        return tf.train.SessionRunArgs({'lrate': self.lrate})

    def after_run(self, run_context, run_values):    
        if self.step < self.num_steps * self.schedule[0]:
            self.learning_rate += self.increment            
        elif self.step < self.num_steps * self.schedule[1]:
            self.learning_rate -= self.decrement
        elif self.step < self.num_steps * self.schedule[2]:
            self.learning_rate -= self.decrement_final
        else:
            if self.should_stop:
                print("Stopped due to learning scheduler hook!")
                run_context.request_stop()
            else:
                return                    
        self.step += 1
        if (self.step % 400 == 0):
            print("Learning scheduler report: Learning rate = {}".format(self.learning_rate))        
        

    def begin(self):
        graph = tf.get_default_graph()
        
        self.lrate = graph.get_tensor_by_name("lrate:0")  
        self.lrate_placeholder = tf.placeholder(tf.float32, [])
        self.assign_op = tf.assign(self.lrate, self.lrate_placeholder)
    
    def end(self, session):
        pass