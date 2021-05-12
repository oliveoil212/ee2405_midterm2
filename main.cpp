#include "mbed.h"
DigitalOut led1(LED1);
DigitalOut led3(LED3);
int threshold_angle = 30;
int mode = -1;
#define UImode 0
#define anglemode 1
#define rpcmode -1
//############## Tensorflow ###########
Thread gesture_detect;
#include "accelerometer_handler.h"
#include "config.h"
#include "magic_wand_model_data.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
constexpr int kTensorArenaSize = 60 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Return the result of the last prediction
int PredictGesture(float* output) {
  // How many times the most recent gesture has been matched in a row
  static int continuous_count = 0;
  // The result of the last prediction
  static int last_predict = -1;

  // Find whichever output has a probability > 0.8 (they sum to 1)
  int this_predict = -1;
  for (int i = 0; i < label_num; i++) {
    if (output[i] > 0.8) this_predict = i;
  }

  // No gesture was detected above the threshold
  if (this_predict == -1) {
    continuous_count = 0;
    last_predict = label_num;
    return label_num;
  }

  if (last_predict == this_predict) {
    continuous_count += 1;
  } else {
    continuous_count = 0;
  }
  last_predict = this_predict;

  // If we haven't yet had enough consecutive matches for this gesture,
  // report a negative result
  if (continuous_count < config.consecutiveInferenceThresholds[this_predict]) {
    return label_num;
  }
  // Otherwise, we've seen a positive result, so clear all our variables
  // and report it
  continuous_count = 0;
  last_predict = -1;

  return this_predict;
}

void _detecting_gesture() {

  // Whether we should clear the buffer next time we fetch data
  bool should_clear_buffer = false;
  bool got_data = false;

  // The gesture index of the prediction
  int gesture_index;

  // Set up logging.
  static tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model = tflite::GetModel(g_magic_wand_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return ;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  static tflite::MicroOpResolver<6> micro_op_resolver;
  micro_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
      tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,
                               tflite::ops::micro::Register_MAX_POOL_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                               tflite::ops::micro::Register_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                               tflite::ops::micro::Register_FULLY_CONNECTED());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                               tflite::ops::micro::Register_SOFTMAX());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,
                               tflite::ops::micro::Register_RESHAPE(), 1);

  // Build an interpreter to run the model with
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  tflite::MicroInterpreter* interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors
  interpreter->AllocateTensors();

  // Obtain pointer to the model's input tensor
  TfLiteTensor* model_input = interpreter->input(0);
  if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != config.seq_length) ||
      (model_input->dims->data[2] != kChannelNumber) ||
      (model_input->type != kTfLiteFloat32)) {
    error_reporter->Report("Bad input tensor parameters in model");
    return ;
  }

  int input_length = model_input->bytes / sizeof(float);

  TfLiteStatus setup_status = SetupAccelerometer(error_reporter);
  if (setup_status != kTfLiteOk) {
    error_reporter->Report("Set up failed\n");
    return ;
  }

  error_reporter->Report("Set up successful...\n");

  while (true) {

    // Attempt to read new data from the accelerometer
    got_data = ReadAccelerometer(error_reporter, model_input->data.f,
                                 input_length, should_clear_buffer);

    // If there was no new data,
    // don't try to clear the buffer again and wait until next time
    if (!got_data) {
      should_clear_buffer = false;
      continue;
    }

    // Run inference, and report any error
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke failed on index: %d\n", begin_index);
      continue;
    }

    // Analyze the results to obtain a prediction
    gesture_index = PredictGesture(interpreter->output(0)->data.f);
    while(mode == anglemode || mode == rpcmode){
        ThisThread::sleep_for(1000ms);
    }
    if(gesture_index == 0) {
        threshold_angle += 10;
        if(threshold_angle > 60) threshold_angle = 30;
    }
    
    // Clear the buffer next time we read data
    should_clear_buffer = gesture_index < label_num;

    // Produce an output
    if (gesture_index < label_num) {
      error_reporter->Report(config.output_message[gesture_index]);
    }
  }
}

//############## Tensorflow ###########
//############## ACC ##################
#include "stm32l475e_iot01_accelero.h"
float angle = 0;
Thread acc;
int16_t vDataXYZ[3] = {0};
int16_t gDataXYZ[3] = {0};
void init_gdata() {
    printf(" initializing the gravity vector....");
    BSP_ACCELERO_Init();
    for(float i = 0; i < 3; i+=0.1) {
        led1 = !led1;
        ThisThread::sleep_for(100ms);
    }
    BSP_ACCELERO_AccGetXYZ(gDataXYZ);
    printf(" done\n");
}
int _updateacc() {
    BSP_ACCELERO_Init();
    for(float i = 0; i < 3; i+=0.1) {
        led1 = !led1;
        ThisThread::sleep_for(100ms);
    }
    BSP_ACCELERO_AccGetXYZ(gDataXYZ);
    while(1) {
        BSP_ACCELERO_AccGetXYZ(vDataXYZ);
        long int dotprodct=0;
        long int normg=0;
        long int normv=0;
        for(int i = 0; i < 3; i++){
            dotprodct += gDataXYZ[i] * vDataXYZ[i];
            normg += gDataXYZ[i] * gDataXYZ[i];
            normv += vDataXYZ[i] * vDataXYZ[i];
        }
        float cosvalue = dotprodct / sqrt(normg) / sqrt (normv);
        angle = acos(cosvalue) * 180 / 3.1415926;
        //   printf("%f\n", angle);
        ThisThread::sleep_for(200ms);
    }
}
//############## ACC ##################
//############## uLCD##################
#include "uLCD_4DGL.h"
uLCD_4DGL uLCD(D1, D0, D2);
Thread monitor;
// using namespace std::chrono;
int _monitor()
{     
    uLCD.reset();
    // // uLCD.background_color(0xDB7093);
    uLCD.cls();
    // uLCD.printf("hw3\nThershold degree\n");
    uLCD.text_width(2); //4X size text
    uLCD.text_height(2);
      uLCD.printf("  30\n");
      uLCD.printf("  40\n");
      uLCD.printf("  50\n");
      uLCD.printf("  60\n");
//       uLCD.text_width(1); //4X size text
//     uLCD.text_height(1);
//     uLCD.printf("Current angle\n");
//   uLCD.text_width(2); //4X size text
//     uLCD.text_height(2);
      while(1){
          if(mode == UImode){
            for(int i = 0; i < 4; i++) {
                uLCD.locate(0,i);
                uLCD.printf(" ");
            } 
            int i = (threshold_angle -30)/10;
            uLCD.locate(0,i);
            uLCD.printf("-");
          }
         uLCD.locate(0,6);
                uLCD.printf(" %.3f", angle);
        //   else{
        //     while(mode == anglemode){
        //         uLCD.locate(0,6);
        //         uLCD.printf("    %f", angle);
        //     }
        //   }
      }
      ThisThread::sleep_for(100ms);
}
//############## uLCD##################
//############## mqtt #################
#include "MQTTNetwork.h"
#include "MQTTmbed.h"
#include "MQTTClient.h"
#include "stm32l475e_iot01_accelero.h"
Thread wholemqtt_thread;
bool threshold_angle_published = false;
// GLOBAL VARIABLES
WiFiInterface *wifi;
InterruptIn btn2(USER_BUTTON);
//InterruptIn btn3(SW3);
volatile int message_num = 0;
volatile int arrivedcount = 0;
volatile bool closed = false;

const char* topic = "Mbed";

Thread mqtt_thread(osPriorityHigh);
EventQueue mqtt_queue;
MQTT::Client<MQTTNetwork, Countdown>* clientptr;
void messageArrived(MQTT::MessageData& md) {
    MQTT::Message &message = md.message;
    char msg[300];
    sprintf(msg, "Message arrived: QoS%d, retained %d, dup %d, packetID %d\r\n", message.qos, message.retained, message.dup, message.id);
        printf(msg);
    ThisThread::sleep_for(1000ms);
    char payload[300];
    sprintf(payload, "Payload %.*s\r\n", message.payloadlen, (char*)message.payload);
        printf(payload);
    if(mode == UImode) {
        threshold_angle_published = true;
        mode = rpcmode;
        return ;
    }
    ++arrivedcount;
    printf("the above is  %d rd message\n", arrivedcount);
    if(arrivedcount >= 10) {
        arrivedcount = 0;
        mode = rpcmode;
        printf("back to RPC mode\n");
        threshold_angle_published = false;

        // closed = true;
    }
}

void publish_threshold_angle(MQTT::Client<MQTTNetwork, Countdown>* client) {
    message_num++;
    MQTT::Message message;
    char buff[100];
    sprintf(buff, "Threshold angle is %d", threshold_angle);
    message.qos = MQTT::QOS0;
    message.retained = false;
    message.dup = false;
    message.payload = (void*) buff;
    message.payloadlen = strlen(buff) + 1;
    int rc = client->publish(topic, message);
    printf("rc:  %d\r\n", rc);
    printf("Puslish message: %s\r\n", buff);
    mode = rpcmode;
    printf("back to RPC mode\n");
}
void publish_current_angle(MQTT::Client<MQTTNetwork, Countdown>* client) {
    message_num++;
    // printf("sdfh\n");
    MQTT::Message message;
    char buff[100];
    sprintf(buff, "angle is %f now", angle);
    message.qos = MQTT::QOS0;
    message.retained = false;
    message.dup = false;
    message.payload = (void*) buff;
    message.payloadlen = strlen(buff) + 1;
    int rc = client->publish(topic, message);
    printf("rc:  %d\r\n", rc);
    printf("Puslish message: %s\r\n", buff);
}

void close_mqtt() {
    closed = true;
}

int _mqtt() {

    wifi = WiFiInterface::get_default_instance();
    if (!wifi) {
        printf("ERROR: No WiFiInterface found.\r\n");
        return -1;
    }


    printf("\nConnecting to %s...\r\n", MBED_CONF_APP_WIFI_SSID);
    int ret = wifi->connect(MBED_CONF_APP_WIFI_SSID, MBED_CONF_APP_WIFI_PASSWORD, NSAPI_SECURITY_WPA_WPA2);
    if (ret != 0) {
        printf("\nConnection error: %d\r\n", ret);
        return -1;
    }


    NetworkInterface* net = wifi;
    MQTTNetwork mqttNetwork(net);
    MQTT::Client<MQTTNetwork, Countdown> client(mqttNetwork);

    //TODO: revise host to your IP
    const char* host = "192.168.80.150";
    printf("Connecting to TCP network...\r\n");

    SocketAddress sockAddr;
    sockAddr.set_ip_address(host);
    sockAddr.set_port(1883);

    printf("address is %s/%d\r\n", (sockAddr.get_ip_address() ? sockAddr.get_ip_address() : "None"),  (sockAddr.get_port() ? sockAddr.get_port() : 0) ); //check setting

    int rc = mqttNetwork.connect(sockAddr);//(host, 1883);
    if (rc != 0) {
        printf("Connection error.");
        return -1;
    }
    printf("Successfully connected!\r\n");

    MQTTPacket_connectData data = MQTTPacket_connectData_initializer;
    data.MQTTVersion = 3;
    data.clientID.cstring = "Mbed";

    if ((rc = client.connect(data)) != 0){
        printf("Fail to connect MQTT\r\n");
    }
    if (client.subscribe(topic, MQTT::QOS0, messageArrived) != 0){
        printf("Fail to subscribe\r\n");
    }
    clientptr = &client;
    mqtt_thread.start(callback(&mqtt_queue, &EventQueue::dispatch_forever));
    btn2.rise(mqtt_queue.event(&publish_threshold_angle, &client));
    //btn3.rise(&close_mqtt);

    // int num = 0;
    // while (num != 1) {
    //     client.yield(100);
    //     ++num;
    // }
    BSP_ACCELERO_Init();
    // int16_t vDataXYZ[3] = {0};
    BSP_ACCELERO_AccGetXYZ(vDataXYZ);
    
    while (1) {
        // if(mode == 0 && threshold_angle_published == false) {
            ThisThread::sleep_for(1000ms);
        //     continue;
        // }
        // ThisThread::sleep_for(100ms);
        // if (closed) break;
        // if (angle < threshold_angle) continue;
        // publish_current_angle(&client);
        // printf("@@@@@@@@@@ mode = %d, thangle = %d, angle = %f\n",mode, threshold_angle, angle);
        // MQTT::Message message;
        // // QoS 0
        // char buf[100];
        // //     BSP_ACCELERO_AccGetXYZ(vDataXYZ);
        // sprintf(buf, "angle = %f", angle);
        // message.qos = MQTT::QOS0;
        // message.retained = false;
        // message.dup = false;
        // message.payload = (void*)buf;
        // message.payloadlen = strlen(buf)+1;
        // //     printf("%d\n",++num);
        // client.publish(topic, message);
        // client.yield(500);
        // publish_threshold_angle(&client);
        if(mode == anglemode && angle >= threshold_angle) {
                publish_current_angle(&client);
             client.yield(500);
        }
    }

    printf("Ready to close MQTT Network......\n");

    if ((rc = client.unsubscribe(topic)) != 0) {
        printf("Failed: rc from unsubscribe was %d\n", rc);
    }
    if ((rc = client.disconnect()) != 0) {
        printf("Failed: rc from disconnect was %d\n", rc);
    }

    mqttNetwork.disconnect();
    printf("Successfully closed!\n");

    return 0;
}
Thread lookingangle;
void _lookingangle(){
    while(1) {
        ThisThread::sleep_for(500ms);
        publish_current_angle(clientptr);
        (*clientptr).yield(500);
        if(mode == anglemode) {
            if(angle > threshold_angle * 1.0) {
                continue;
            }
        }
    }
}
//############## mqtt #################
//############## RPC ##################
#include "mbed_rpc.h"

Thread rpc;
BufferedSerial pc(USBTX, USBRX);
void modeControl(Arguments *in, Reply *out);
RPCFunction moderpc(&modeControl, "mode");
void modeControl(Arguments *in, Reply *out) {
    mode = in->getArg<int>();
    led3 = mode; // orange for angle, blue for gestureUI
    arrivedcount = 0;
    
    if(mode == anglemode) {
        out->putData("tilt angle detection mode");
        init_gdata();
    }
    else if(mode == UImode) {
        out->putData("gesture UI mode");
        threshold_angle_published = false;
    }
    else out->putData("please input either /mode/run 1 for angle dectecting or /mode/run 0 for gestureUI");
}
void gesture_UI(Arguments *in, Reply *out){
    mode = UImode;
    led3 = mode; // orange for angle, blue for gestureUI
    arrivedcount = 0;
    out->putData("gesture UI mode");
    threshold_angle_published = false;
}
RPCFunction rpc_gesture_UI(&gesture_UI, "UI");
void tilt_angle_detection(Arguments *in, Reply *out){
    mode = anglemode;
    led3 = mode; // orange for angle, blue for gestureUI
    arrivedcount = 0;
    out->putData("tilt angle detection mode");
    init_gdata();
}
RPCFunction  rpc_tilt_angle_detection(&tilt_angle_detection, "detect");
void _rpcio(){
    char buf[256], outbuf[256];

    FILE *devin = fdopen(&pc, "r");
    FILE *devout = fdopen(&pc, "w");

    while(1) {
        memset(buf, 0, 256);
        for (int i = 0; ; i++) {
            char recv = fgetc(devin);
            if (recv == '\n') {
                printf("\r\n");
                break;
            }
            buf[i] = fputc(recv, devout);
        }
        //Call the static call method on the RPC class
        RPC::call(buf, outbuf);
        printf("%s\r\n", outbuf);
    }
}
//############## RPC ##################


int main () {
    rpc.start(_rpcio);
    acc.start(_updateacc);
    monitor.start(_monitor);
    wholemqtt_thread.start(_mqtt);
    gesture_detect.start(_detecting_gesture);
    // lookingangle.start(_lookingangle);
    while(1){}
    
}
