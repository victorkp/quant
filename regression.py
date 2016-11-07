import tensorflow as tf
import csv
import tempfile

tf.logging.set_verbosity(tf.logging.ERROR)

def build_estimator():
    gdp_high = tf.contrib.layers.real_valued_column("gdp_high")
    gdp_change = tf.contrib.layers.real_valued_column("gdp_change")

    manufacturer_high = tf.contrib.layers.real_valued_column("manufacture_orders_high")
    manufacturer_change = tf.contrib.layers.real_valued_column("manufacture_orders_change")

    manufacturer_durable_high = tf.contrib.layers.real_valued_column("manufacture_durable_orders_high")
    manufacturer_durable_change = tf.contrib.layers.real_valued_column("manufacture_durable_orders_change")

    spy_change = tf.contrib.layers.real_valued_column("spy_change")
    spy_rsi_3  = tf.contrib.layers.real_valued_column("spy_rsi_3")
    spy_rsi_10 = tf.contrib.layers.real_valued_column("spy_rsi_10")
    spy_rsi_14 = tf.contrib.layers.real_valued_column("spy_rsi_14")
    spy_average_10 = tf.contrib.layers.real_valued_column("spy_average_10")
    spy_average_20 = tf.contrib.layers.real_valued_column("spy_average_20")
    spy_average_40 = tf.contrib.layers.real_valued_column("spy_average_40")
    spy_average_80 = tf.contrib.layers.real_valued_column("spy_average_80")

    columns = [gdp_high, gdp_change,
               manufacturer_high, manufacturer_change,
               manufacturer_durable_high, manufacturer_durable_change,
               spy_change,
               spy_rsi_3, spy_rsi_10, spy_rsi_14,
               spy_average_10, spy_average_20, spy_average_40, spy_average_80];

    m = tf.contrib.learn.LinearRegressor(model_dir="model",
                                         feature_columns=columns,
                                         enable_centered_bias=True,
                                         optimizer=tf.train.FtrlOptimizer(
                                                 learning_rate=0.1,
                                                 l1_regularization_strength=1.0,
                                                 l2_regularization_strength=1.0)
                                        )

    return m

def input_fn(start, end):
    input_data = list(csv.reader(open("data/market.csv")))
    input_index = 1

    gdp_high = []
    gdp_change = []

    manufacturer_high = []
    manufacturer_change = []

    manufacturer_durable_high = []
    manufacturer_durable_change = []

    spy_change = []

    spy_rsi_3  = []
    spy_rsi_10 = []
    spy_rsi_14 = []
    spy_average_10 = []
    spy_average_20 = []
    spy_average_40 = []
    spy_average_80 = []

    labels = []

    for index in range(start, end):
        gdp_high.append(float(input_data[index][1]));
        gdp_change.append(float(input_data[index][2]));

        manufacturer_high.append(float(input_data[index][3]))
        manufacturer_change.append(float(input_data[index][4]))

        manufacturer_durable_high.append(float(input_data[index][5]))
        manufacturer_durable_change.append(float(input_data[index][6]))

        spy_change.append(float(input_data[index][7]))
        #spy_change.append(0)

        spy_rsi_3.append(float(input_data[index][8]))
        spy_rsi_10.append(float(input_data[index][9]))
        spy_rsi_14.append(float(input_data[index][9]))
        #spy_rsi_3.append(0)
        #spy_rsi_10.append(0)
        #spy_rsi_14.append(0)

        spy_average_10.append(float(input_data[index][10])) 
        spy_average_20.append(float(input_data[index][11])) 
        spy_average_40.append(float(input_data[index][12])) 
        spy_average_80.append(float(input_data[index][13])) 
        #spy_average_10.append(0) 
        #spy_average_20.append(0) 
        #spy_average_40.append(0) 
        #spy_average_80.append(0) 

        labels.append(float(input_data[index+1][7]))

        # if(float(input_data[index+1][7]) >= 0):
        #     labels.append(1)
        # else:
        #     labels.append(0)

    columns = {
                "gdp_high" : tf.constant(gdp_high),
                "gdp_change" : tf.constant(gdp_change),

                "manufacture_orders_high" : tf.constant(manufacturer_high),
                "manufacture_orders_change" : tf.constant(manufacturer_change),

                "manufacture_durable_orders_high" : tf.constant(manufacturer_durable_high),
                "manufacture_durable_orders_change" : tf.constant(manufacturer_durable_change),

                "spy_change" : tf.constant(spy_change),

                "spy_rsi_3" : tf.constant(spy_rsi_3),
                "spy_rsi_10" : tf.constant(spy_rsi_10),
                "spy_rsi_14" : tf.constant(spy_rsi_14),

                "spy_average_10" : tf.constant(spy_average_10),
                "spy_average_20" : tf.constant(spy_average_20),
                "spy_average_40" : tf.constant(spy_average_40),
                "spy_average_80" : tf.constant(spy_average_80)
              }
    
    return columns, tf.constant(labels)


meta_results = []

multiplier = 10000
for train_steps in range(multiplier, 100 * multiplier, multiplier):
    print
    print
    print "Training Steps: %d" % (train_steps)

    m = build_estimator()
    m.fit(input_fn=lambda: input_fn(2, 1600), steps=train_steps)

    results = m.evaluate(input_fn=lambda: input_fn(2, 1600), steps=1)
    print "Train Loss: %s, averaged to %s" % (results["loss"], results["loss"] / 1598.0)
    train_loss = results["loss"]

    results = m.evaluate(input_fn=lambda: input_fn(1600, 2206), steps=1)
    print "Test Loss: %s, averaged to %s" % (results["loss"], results["loss"] / 606.0)
    test_loss = results["loss"]

    meta_results.append((train_steps, train_loss, test_loss))

print
print

meta_results.sort(key = lambda x: x[2])
for result in meta_results:
    print "%d\t%s\t%s" % (result[0], result[1], result[2])

#for key in sorted(results):
#    print("%s: %s" % (key, results[key]))


# print m.get_variable_value("gdp_high")
# print m.get_variable_value("gdp_change")
# print m.get_variable_value("manufacture_orders_high")
# print m.get_variable_value("manufacture_orders_change")
# print m.get_variable_value("manufacture_durable_orders_high")
# print m.get_variable_value("manufacture_durable_orders_change")
# print m.get_variable_value("spy_change")
# print m.get_variable_value("spy_rsi_3")
# print m.get_variable_value("spy_rsi_10")
# print m.get_variable_value("spy_rsi_14")
# print m.get_variable_value("spy_average_10")
# print m.get_variable_value("spy_average_20")
# print m.get_variable_value("spy_average_40")
# print m.get_variable_value("spy_average_80")


