from fcn_vgg import Net

if __name__ == "__main__":

    net = Net(is_training=True)
    net.train_net()

    net = Net(is_training=False)
    net.run_test(num=16)
