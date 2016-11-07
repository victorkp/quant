import tensorflow as tf
import csv
import tempfile

# old_spy_price = observation[8]
# old_gld_price = observation[11]
# observation = input_data[input_index][1:] # Ignore timestamp, don't want that in weights
# spy_profit = (observation[8] - old_spy_price)/(old_spy_price)
# gld_profit = (observation[11] - old_gld_price)/(old_gld_price)
# done = input_index == 460

def build_estimator():
    gdp_change = tf.contrib.layers.real_valued_column("gdp_change")
    manufacturer_change = tf.contrib.layers.real_valued_column("manufacture_orders_change")
    manufacturer_durable_change = tf.contrib.layers.real_valued_column("manufacture_durable_orders_change")
    spy_change = tf.contrib.layers.real_valued_column("spy_change")
    slv_change = tf.contrib.layers.real_valued_column("slv_change")
    gld_change = tf.contrib.layers.real_valued_column("gld_change")

    columns = [gdp_change, manufacturer_change, manufacturer_durable_change,
               spy_change, slv_change, gld_change];

    m = tf.contrib.learn.LinearClassifier(model_dir="model", feature_columns=columns)

    return m

def input_fn():
    input_data = list(csv.reader(open("data/market.csv")))
    input_index = 1

    gdp_change = []
    manufacturer_change = []
    manufacturer_durable_change = []
    spy_change = []
    slv_change = []
    gld_change = []

    labels     = []

    last_spy_price = float(input_data[1][8])
    last_gld_price = float(input_data[1][11])
    last_slv_price = float(input_data[1][14])

    for index in range(2, 459):
        gdp_change.append(float(input_data[index][2]))
        manufacturer_change.append(float(input_data[index][4]))
        manufacturer_durable_change.append(float(input_data[index][6]))
        spy_change.append((float(input_data[index][8])  - last_spy_price)/(last_spy_price))
        gld_change.append((float(input_data[index][11]) - last_gld_price)/(last_gld_price))
        slv_change.append((float(input_data[index][14]) - last_slv_price)/(last_slv_price))

        slv_change.append((float(input_data[index][14]) - last_slv_price)/(last_slv_price))

        last_spy_price = float(input_data[index][8])
        last_gld_price = float(input_data[index][11])
        last_slv_price = float(input_data[index][14])

        next_spy_change = (float(input_data[index+1][8])  - last_spy_price)/last_spy_price
        next_gld_change = (float(input_data[index+1][11]) - last_gld_price)/last_gld_price

        if next_spy_change == 0.0:
            next_spy_change == 0.0001

        if next_gld_change == 0.0:
            next_gld_change == 0.0001

        print "%s %s" % (next_gld_change, next_spy_change)
        if next_spy_change > 0 and next_gld_change > 0:
            labels.append(min(1, next_spy_change / next_gld_change))
        elif next_spy_change > 0:
            labels.append(1.0)
        elif next_gld_change > 0:
            labels.append(0.001)
        else:
            labels.append(min(1.0, next_gld_change / next_spy_change))

    columns = {
                "gdp_change" : tf.constant(gdp_change),
                "manufacture_orders_change" : tf.constant(manufacturer_change),
                "manufacture_durable_orders_change" : tf.constant(manufacturer_durable_change),
                "spy_change" : tf.constant(spy_change),
                "gld_change" : tf.constant(gld_change),
                "slv_change" : tf.constant(slv_change)
              }
    
    return columns, tf.constant(labels)


m = build_estimator()
m.fit(input_fn=lambda: input_fn(), steps=1)
results = m.evaluate(input_fn=lambda: input_fn(), steps=1)

for key in sorted(results):
    print("%s: %s" % (key, results[key]))

