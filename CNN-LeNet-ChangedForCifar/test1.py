demos = {
    "layer_50":[{"depth": 256,"num_class": 3},
                {"depth": 512,"num_class": 4},
                {"depth": 1024,"num_class": 6},
                {"depth": 2048,"num_class": 3}],

    "layer_101": [{"depth": 256, "num_class": 3},
                  {"depth": 512, "num_class": 4},
                  {"depth": 1024, "num_class": 23},
                  {"depth": 2048, "num_class": 3}],

    "layer_152": [{"depth": 256, "num_class": 3},
                  {"depth": 512, "num_class": 8},
                  {"depth": 1024, "num_class": 36},
                  {"depth": 2048, "num_class": 3}]
               }

if 5>1:
    demo_num = 0
    for demo in demos["layer_152"]:
        demo_num += 1
        print("--------------------------------------------")
            #堆叠子类瓶颈模块
        print("num_" + str(demo_num))
        for i in range(int(demo["num_class"])):
            print("demo_num:  "+str(demo_num))
            if demo_num is not 4:
                if i == int(demo["num_class"]) - 1:
                    stride = 2
                else:
                    stride = 1
            else:
                stride = 1
            print("bottleneck_" + str(i + 1))
        print("--------------------------------------------")
