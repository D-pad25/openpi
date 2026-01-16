#include <AccelStepper.h>
#include <Encoder.h>
#include <DynamixelShield.h>
#include <ros.h>
#include <std_msgs/Empty.h>
#include <std_msgs/String.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Header.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Int16.h>

/// STEPPER ////
#define EN A11      /* Enable pin for all stepper outputs */

#define X_DIR  A15   /* Direction pin for X axis */
#define X_STEP A12  /* Step pin for X axis */

#define Y_DIR A16   /* Direction pin for Y axis */
#define Y_STEP A13  /* Step pin for Y axis */

#define Z_DIR A17   /* Direction pin for Z axis */
#define Z_STEP A14  /* Step pin for Z axis */
/// STEPPER ////

// #define MIN_LIFT_POSITION_METRES -0.78
// #define MAX_LIFT_POSITION_METRES 1.0

// #define MIN_HORIZONTAL_POSITION_METRES -0.77
// #define MAX_HORIZONTAL_POSITION_METRES 0.8

#define MIN_LIFT_POSITION_METRES 0.5
#define MAX_LIFT_POSITION_METRES 1.0

#define MIN_HORIZONTAL_POSITION_METRES 0.19
#define MAX_HORIZONTAL_POSITION_METRES 0.8

#define MM_PER_REV 10.0
#define METRES_PER_REV 0.01

#define REV_PER_MM = 0.1
#define REV_PER_METRE = 100

#define MICRO_PER_STEP 4.0 //32
#define STEPS_PER_MM (20.0*MICRO_PER_STEP)
#define STEPS_PER_MM_HORIZONTAL ((200*MICRO_PER_STEP)/(3*20))  //TBC


#define STEPS_PER_REV 200.0*MICRO_PER_STEP

#define COUNTS_PER_REV 1024

#define METRES_TO_STEPS_HORIZONTAL(metres) (metres*1000.0*STEPS_PER_MM_HORIZONTAL)
#define STEPS_TO_METRES_HORIZONTAL(steps) (steps/1000.0/STEPS_PER_MM_HORIZONTAL)

#define METRES_TO_STEPS(metres) (metres*1000.0*STEPS_PER_MM)
#define STEPS_TO_METRES(steps) (steps/1000.0/STEPS_PER_MM)

//Converters between counts and millimeters for encoders (vertical)
#define COUNTS_TO_METRES(counts) (counts*MM_PER_REV/COUNTS_PER_REV/1000.0)
#define METRES_TO_COUNTS(metres) (metres*COUNTS_PER_REV/MM_PER_REV/1000.0)

//Converters for counts and steps
//#define COUNT_TO_STEPS(counts) (count*STEPS_PER_REV/COUNTS_PER_REV)
//#define STEPS_TO_COUNT(steps) (steps*COUNTS_PER_REV/STEPS_PER_REV)


#define JOINT_FEEDBACK_LEN 5
#define ESTOP_PIN   4
#define PUBLISHPERIOD 100 // in ms
#define HEARTBEATPERIOD 10
#define STATUSRATE  1
//
#define USE_TEENSY_HW_SERIAL
//#define USE_USBCON/

#define PI 3.1415926535897932384626433832795

#define encoderPin0 22
#define encoderPin1 12

#define ENCMOTTHRESH 100000
#define ENCTHRESH 60000

//----------------------------------------Pruning Tool Definitions
#define ADDR_MX_CW_ANGLE_LIMIT 6
#define ADDR_MX_CCW_ANGLE_LIMIT 8
#define GOAL_SPEED  32 
#define ADDR_MX_TORQUE_ENABLE 24
#define ADDR_MX_GOAL_POSITION 30
#define ADDR_MX_PRESENT_POSITION 36
#define ADDR_MX_MOVING_SPEED 32
#define ADDR_MX_CW_ANGLE_LIMIT 6
#define ADDR_MX_CCW_ANGLE_LIMIT 8
#define ADDR_MX_TORQUE_LIMIT 34 


#if defined(ARDUINO_AVR_UNO) || defined(ARDUINO_AVR_MEGA2560)
#include <SoftwareSerial.h>
SoftwareSerial soft_serial(7, 8); // DYNAMIXELShield UART RX/TX
#define DEBUG_SERIAL soft_serial
#elif defined(ARDUINO_SAM_DUE) || defined(ARDUINO_SAM_ZERO)
#define DEBUG_SERIAL SerialUSB
#else
#define DEBUG_SERIAL Serial
#endif

//strawberry gripper ID = 1
//const uint8_t DXL_MOTOR_GRIPPER_ID = 1;

// Pruning Tool Prune ID = 101
//const uint8_t DXL_MOTOR_PRUNE_ID = 101;

// If the motor is the XC330
const uint8_t DXL_MOTOR_PRUNE_ID = 3;

//tomato gripper ID = 0
const uint8_t DXL_MOTOR_GRIPPER_ID = 1;

// const float DXL_PROTOCOL_VERSION = 2.0;
const float DXL_PROTOCOL_VERSION = 1.0;

const float XC_PRUNE_DXL_PROTOCOL_VERSION = 2.0;
int dxl_dir_pin = A1;

DynamixelShield dxl(Serial2, dxl_dir_pin);

/// STEPPER
AccelStepper horizontalAxis(1, X_STEP, X_DIR);
AccelStepper liftAxis1(1, Y_STEP, Y_DIR);
AccelStepper liftAxis2(1, Z_STEP, Z_DIR);

// ENCODERS
// ENCODERS
Encoder encoderHoriz(21, 22);
Encoder encoderVert1(18, 19);
Encoder encoderVert2(28, 29); // Horizontal 28,29


#define encoderHorizPin 21
#define encoderVertPin1 18
#define encoderVertPin2 28


unsigned long startMillis, currentMillis, heartbeatMillis, publishMillis ;  //some global variables available anywhere in the program
int led = 13;
const byte xLimitPin = 33;
const byte yLimitPin = 35;
const byte zLimitPin = 36;
volatile long positionEncoderHoriz  = 0;
volatile long positionEncoderVert1 = 0;
volatile long positionEncoderVert2 = 0;
volatile long newEncoderHorizPos, newEncoderVert1Pos, newEncoderVert2Pos;

volatile bool eStop = false;
volatile bool motorRunning = false;
volatile int enc0DiffCounter, enc1DiffCounter = 0;
volatile bool homingFinished = false;

//ros::NodeHandle_<ArduinoHardware, 4, 4, 1024, 1024> nh;
ros::NodeHandle_<ArduinoHardware, 8, 8, 2048, 2048> nh;
std_msgs::Float32MultiArray motor_states;
std_msgs::Float32 homing_state;

float motor_data_array[JOINT_FEEDBACK_LEN];

sensor_msgs::JointState motor_commands;

// --------- Global Tool Vars ---------
int Gripper = 1;
int enableGripper = 1;
int enablePrune = 1;

//from dynamixel to ros
float wrap_2pi(float degree) {

  if (degree > 180) {
    return (degree - 360);
  } else {
    return degree;
  }
}

//from ros to dynamixel
float unwrap_2pi(float degree) {

  if (degree < 0) {
    return (degree + 360);
  } else {
    return degree;
  }
}


//Callback to receive joint states from PC via ROS message
void motor_commandCb(const sensor_msgs::JointState& motor_command) {

  float shoulder_joint_pos = 0;
  float elbow_joint_pos = 0;
  float wrist_joint_1_pos = 0;
  double liftPosition = 0;
  double horizontalPosition = 0;

  //nh.loginfo("callback triggered");

  if (!homingFinished) {
    return;
  }

  liftPosition = METRES_TO_STEPS((double)(motor_command.position[0]));
  horizontalPosition = METRES_TO_STEPS_HORIZONTAL((double)(motor_command.position[1]));


  //  long liftVelocity = (long)(motor_command.velocity[0]*1000*32*20);/
  float liftVelocity = 10;
  //  long horizontVelocity = (/long)(motor_command.velocity[1]*1000*32*20);
  float horizontVelocity = 100;

  liftAxis1.moveTo(liftPosition);
  liftAxis1.setMaxSpeed(1000);
  liftAxis1.setAcceleration(5000);

  liftAxis2.moveTo(liftPosition);
  liftAxis2.setMaxSpeed(1000);
  liftAxis2.setAcceleration(5000);

  horizontalAxis.moveTo(horizontalPosition);
  horizontalAxis.setMaxSpeed(2000);
  horizontalAxis.setAcceleration(5000);

  motor_states.data[2] = motor_command.position[2];
  motor_states.data[3] = motor_command.position[3];
  motor_states.data[4] = motor_command.position[4];

  //nh.loginfo("finished motor command CB");/
  //Some code to check the desired/target position against where AccelStepper thinks it is
  //  char VT1_msg[32];
  //  char str_temp[6];
  //  dtostrf(shoulder_joint_pos, 4, 2, str_temp);
  ////  /sprintf(VT1_msg,"%s F", str_temp);
  //  sprintf(VT1_msg,"Vertical TARGET position: %f", shoulder_joint_pos);
  //  nh.loginfo(VT1_msg);


}


//Callback to receive gripper open and close commands from PC via ROS message
// void gripper_commandCb(const std_msgs::Bool& gripper_command) {

//   // //tomato gripper angles
//   // float open_angle = 120;
//   // float close_angle = 359;


//   //tomato gripper angles
//   float open_angle = 5;
//   float close_angle = 179;

//   //strawberry gripper angles
// //  float open_angle = 175;//
// //  float close_angle = 195;/

//   if(gripper_command.data == true){
//     //close gripper
//     dxl.setGoalPosition(DXL_MOTOR_GRIPPER_ID, close_angle, UNIT_DEGREE);

//   }else if(gripper_command.data == false){
//     //open gripper
//     dxl.setGoalPosition(DXL_MOTOR_GRIPPER_ID, open_angle, UNIT_DEGREE);

//   }

// }

//Callback to receive gripper open and close commands from PC via ROS message
void gripper_commandCb(const std_msgs::Int16& gripper_command) {

  // //tomato gripper angles
  // float open_angle = 120;
  // float close_angle = 359;


  //strawberry gripper angles
  //  float open_angle = 175;//
  //  float close_angle = 195;/
//  char log_msg4[64];
//  sprintf(log_msg4, "Gripper Command received: %d", gripper_command.data);
//  nh.loginfo(log_msg4);
  if (enableGripper) {
    dxl.setGoalPosition(DXL_MOTOR_GRIPPER_ID, (float)gripper_command.data, UNIT_DEGREE);
  }
}

// Callback to receive gripper open and close commands from PC via ROS message
//void gripper_commandCb(const std_msgs::Int16& gripper_command) {
//  char log_msg[128];
//
//  // Log raw incoming value
//  sprintf(log_msg, "[DEBUG] Gripper Command received: %d", gripper_command.data);
//  nh.loginfo(log_msg);
//
//  // Check gripper enable flag
//  if (!enableGripper) {
//    nh.logwarn("[DEBUG] Gripper is DISABLED. Command ignored.");
//    return;
//  }
//
//  nh.loginfo("[DEBUG] Gripper is ENABLED. Sending to Dynamixel...");
//
//  // Log before sending to motor
//  sprintf(log_msg, "[DEBUG] setGoalPosition -> ID:%d, Angle:%d deg",
//          DXL_MOTOR_GRIPPER_ID, gripper_command.data);
//  nh.loginfo(log_msg);
//
//  // Try sending command
//  int result = dxl.setGoalPosition(DXL_MOTOR_GRIPPER_ID,
//                                   (float)gripper_command.data,
//                                   UNIT_DEGREE);
//
//  // Check return code if API provides one
//  if (result != 0) {
//    sprintf(log_msg, "[ERROR] setGoalPosition failed with code %d", result);
//    nh.logerror(log_msg);
//  } else {
//    nh.loginfo("[DEBUG] setGoalPosition success!");
//  }
//}


void setupGripper() {
  /// DXL MOTOR SETUP///
  dxl.setPortProtocolVersion(DXL_PROTOCOL_VERSION);
  // dxl.ping(DXL_MOTOR_GRIPPER_ID);

  dxl.ping(DXL_MOTOR_GRIPPER_ID);
  
  //  dxl.torqueOff(DXL_MOTOR_GRIPPER_ID);/
  //  dxl.setOperatingMode(DXL_MOTOR_GRIPPER_ID, OP_POSITION);/
  dxl.torqueOn(DXL_MOTOR_GRIPPER_ID);

}

void gripper_enableCb(const std_msgs::Int16& enable_cmd) {

  if (!enableGripper && enable_cmd.data) {
    setupGripper();
  }

  enableGripper = enable_cmd.data;

}


void homing_cb(const std_msgs::Float32& homing_command) {
  nh.loginfo("Initializing homing sequence");
  homingSequence();
  nh.loginfo("Finished homing sequence");
  publishHomingStatus();
  homingFinished = true;
}

//These are ROSSERIAL publisher and subscriber object creation

ros::Publisher motor_states_pub("stepper_joint_feedback", &motor_states);
ros::Publisher homing_pub("homing_feedback", &homing_state);

ros::Subscriber<sensor_msgs::JointState> motor_commands_sub("/motor_command", &motor_commandCb);
ros::Subscriber<std_msgs::Float32> homing_commands_sub("/stepper_homing_command", &homing_cb);

// ------------------ GTRIPPER TOOL ------------------
// Create gripper publisher (for publishing position) and subscriber (for gripper command)
std_msgs::Float32 gripper_pos_msg;
ros::Publisher gripper_pose_pub("gripper_position", &gripper_pos_msg);
ros::Subscriber<std_msgs::Int16> gripper_commands_sub("/gripper_command", &gripper_commandCb);
ros::Subscriber<std_msgs::Int16> gripper_enable_sub("/gripper_enable", &gripper_enableCb);



void getGripperPosition()
{
  float gripper_pos = dxl.getPresentPosition(DXL_MOTOR_GRIPPER_ID, UNIT_DEGREE);

  std_msgs::Float32 msg;
  msg.data = gripper_pos;

  // nh.loginfo("Publishing Homing Status");
  gripper_pose_pub.publish(&msg);
}

// ------------------ PRUNING TOOL ------------------
// Create pruning publisher (for publishing position) and subscriber (for prune command)
std_msgs::Float32 prune_speed_msg;
ros::Publisher prune_pose_pub("prune_position", &prune_speed_msg);
ros::Subscriber<std_msgs::Int16> prune_commands_sub("/prune_command", &prune_commandCb);
ros::Subscriber<std_msgs::Int16> prune_enable_sub("/prune_enable", &prune_enableCb);

// ----- Prune Setup Function -----
// If using AX-12A
//void setupPrune() {
//  /// DXL MOTOR SETUP///
//
//  dxl.setPortProtocolVersion(DXL_PROTOCOL_VERSION);
//  dxl.ping(DXL_MOTOR_PRUNE_ID);
//
//  dxl.torqueOff(DXL_MOTOR_PRUNE_ID);
//
//  // Wheel mode on AX series: BOTH limits = 0 (ticks)
//  dxl.writeControlTableItem(ControlTableItem::BAUD_RATE, DXL_MOTOR_PRUNE_ID, 34);
//  
//  dxl.writeControlTableItem(ControlTableItem::CW_ANGLE_LIMIT,  DXL_MOTOR_PRUNE_ID, 0);
//  dxl.writeControlTableItem(ControlTableItem::CCW_ANGLE_LIMIT, DXL_MOTOR_PRUNE_ID, 0);
//
//  // Optional safety caps
//  dxl.writeControlTableItem(ControlTableItem::TORQUE_LIMIT, DXL_MOTOR_PRUNE_ID, 800);
//
//  // Start at stop
//  dxl.writeControlTableItem(ControlTableItem::MOVING_SPEED, DXL_MOTOR_PRUNE_ID, 0);
//
//  dxl.torqueOn(DXL_MOTOR_PRUNE_ID);
//}

//If Using XC330
void setupPrune() {
  /// DXL MOTOR SETUP///

  dxl.setPortProtocolVersion(XC_PRUNE_DXL_PROTOCOL_VERSION);
  dxl.ping(DXL_MOTOR_PRUNE_ID);

  dxl.torqueOff(DXL_MOTOR_PRUNE_ID);

  // Wheel mode on AX series: BOTH limits = 0 (ticks)
  dxl.writeControlTableItem(ControlTableItem::BAUD_RATE, DXL_MOTOR_PRUNE_ID, 1);
  

  // Set to Velocity Control Mode (Operating Mode = 1)
  dxl.writeControlTableItem(ControlTableItem::OPERATING_MODE, DXL_MOTOR_PRUNE_ID, 1);

  // Optional: limit torque (Current Limit)
  dxl.writeControlTableItem(ControlTableItem::CURRENT_LIMIT, DXL_MOTOR_PRUNE_ID, 800); // adjust as needed

  // Start at zero velocity
  dxl.writeControlTableItem(ControlTableItem::GOAL_VELOCITY, DXL_MOTOR_PRUNE_ID, 0);

  dxl.torqueOn(DXL_MOTOR_PRUNE_ID);
}


// ----- Prune Position Function -----
void getPrunePosition()
{
  //float prune_pos_msg = dxl.writeControlTableItem(GOAL_SPEED, DXL_MOTOR_PRUNE_ID);
  float prune_pos = dxl.getPresentPosition(DXL_MOTOR_PRUNE_ID, UNIT_DEGREE);
  std_msgs::Float32 msg;
  msg.data = prune_pos;

  // Publish message
  prune_pose_pub.publish(&msg);
}

// ----- Prune Command Callback Function -----
//Callback to receive pruning commands from PC via ROS message
typedef enum{
  DEFAULT,
  STOP,
  CLOCKWISE,
  COUNTERCLOCKWISE,
}System_State;

System_State current_state = DEFAULT;

//void prune_commandCb(const std_msgs::Int16& prune_command){
//  // Only send commands if tool is enabled
//  //if (enablePrune) {
////    int target_position = prune_command.data;  // Expect angle in degrees (0-300)
//////    // Send target position to Dynamixel
////    dxl.setGoalPosition(DXL_MOTOR_PRUNE_ID, target_position, UNIT_DEGREE);
////  }
//  uint16_t prune_speed = 1023;
//  if (!enablePrune) return;
//
//  // Expect -1023..+1023, sign = direction (AX-12A bit10)
//  int16_t s = prune_command.data;
////  if (s >  1023) s =  1023;
////  if (s < -1023) s = -1023;
////
////  // Small deadband so it actually turns
////  if (s > 0 && s < 20)   s = 20;
////  if (s < 0 && s > -20)  s = -20;
//
//  uint16_t reg = 0;
//
//  switch(s){
//    case 0:
//      current_state = STOP;
//      reg = 0;
//      break;
//    case 1:
//      current_state = CLOCKWISE;
//      reg = prune_speed;
//      break;
//    case 2:
//      current_state = COUNTERCLOCKWISE;
//      reg = prune_speed | (1u << 10);
//      break;
//    default:
//      current_state = STOP;
//      reg = 0;
//      break;
//  }
//dxl.writeControlTableItem(ControlTableItem::MOVING_SPEED, DXL_MOTOR_PRUNE_ID, reg);
//  
//}

//Use if the prune with the xc330 is on there
void prune_commandCb(const std_msgs::Int16& prune_command)
{
  if (!enablePrune) return;

  int16_t s = prune_command.data;  // Expect command: 0=stop, 1=CW, 2=CCW
  int32_t goal_velocity = 0;

  switch (s) {
    case 0:  // stop
      current_state = STOP;
      goal_velocity = 0;
      break;
    case 1:  // clockwise
      current_state = CLOCKWISE;
      goal_velocity = -200;  // negative = CW (you can tune magnitude)
      break;
    case 2:  // counterclockwise
      current_state = COUNTERCLOCKWISE;
      goal_velocity = 200;   // positive = CCW
      break;
    case 3:
          // --- Go to default position ---
      dxl.torqueOff(DXL_MOTOR_PRUNE_ID);
      dxl.writeControlTableItem(ControlTableItem::OPERATING_MODE, DXL_MOTOR_PRUNE_ID, 3); // Position Control Mode
      dxl.torqueOn(DXL_MOTOR_PRUNE_ID);

      const float DEFAULT_POSITION = 270.0;  // degrees
      dxl.setGoalPosition(DXL_MOTOR_PRUNE_ID, DEFAULT_POSITION, UNIT_DEGREE);

      // Wait until the motor reaches position
      while (fabs(dxl.getPresentPosition(DXL_MOTOR_PRUNE_ID, UNIT_DEGREE) - DEFAULT_POSITION) > 1.0) {
        nh.spinOnce();  // keep rosserial alive
        delay(50);
      }

      // Switch back to Velocity Mode
      dxl.torqueOff(DXL_MOTOR_PRUNE_ID);
      dxl.writeControlTableItem(ControlTableItem::OPERATING_MODE, DXL_MOTOR_PRUNE_ID, 1); // Velocity Mode
      dxl.torqueOn(DXL_MOTOR_PRUNE_ID);

      // Stop velocity
      dxl.writeControlTableItem(ControlTableItem::GOAL_VELOCITY, DXL_MOTOR_PRUNE_ID, 0);
      return;  // exit early (we're done)
      break;
    default:
      current_state = STOP;
      goal_velocity = 0;
      break;
  }

  // Write goal velocity (signed 4 bytes)
  dxl.writeControlTableItem(ControlTableItem::GOAL_VELOCITY, DXL_MOTOR_PRUNE_ID, goal_velocity);
}


// ----- Gripper Enable Callback Function -----
void prune_enableCb(const std_msgs::Int16& enable_cmd) {

  if (!enablePrune && enable_cmd.data) {
    setupPrune();
  }
if (!enable_cmd.data) dxl.writeControlTableItem(ADDR_MX_MOVING_SPEED, DXL_MOTOR_PRUNE_ID, 0);
  enablePrune = enable_cmd.data;

}

void setup() {
  pinMode(led, OUTPUT);

  /// STEPPER MOTOR ///
  // driver boards on PCB
  pinMode(EN, OUTPUT);
  digitalWrite(EN, LOW);

  //create Accelstepper motor objects
  horizontalAxis.setMaxSpeed(50000);
  liftAxis1.setMaxSpeed(50000); //25600
  liftAxis2.setMaxSpeed(50000); //25600

  liftAxis1.setAcceleration(1000);
  liftAxis2.setAcceleration(1000);
  horizontalAxis.setAcceleration(1000);

  liftAxis1.setPinsInverted(false, false, false);
  liftAxis2.setPinsInverted(false, false, false);
  horizontalAxis.setPinsInverted(false, false, false);

  //Pin setup for stepper limit switches
  pinMode(xLimitPin, INPUT_PULLUP);
  pinMode(yLimitPin, INPUT_PULLUP);
  pinMode(zLimitPin, INPUT_PULLUP);

  /// DXL MOTOR SETUP///
  //dxl.begin(57600); //Default: 115200
  //dxl.begin(100000);


  // normal ops
  dxl.begin(57600);
  setupGripper();
  setupPrune();

  // Initialise node
  nh.initNode();
  

  // Advertise to nodes
  nh.advertise(motor_states_pub);
  nh.advertise(homing_pub);
  nh.advertise(gripper_pose_pub);
  nh.advertise(prune_pose_pub);

  // Subscribe nodes
  nh.subscribe(motor_commands_sub);
  nh.subscribe(homing_commands_sub);
  nh.subscribe(gripper_commands_sub);
  nh.subscribe(gripper_enable_sub);
  nh.subscribe(prune_commands_sub);
  nh.subscribe(prune_enable_sub);

  motor_states.data_length = JOINT_FEEDBACK_LEN;  // define the length of the array
  motor_states.data = motor_data_array;
  homing_state.data = 1;

  startMillis = millis();  //initial start times
  heartbeatMillis = millis();
  publishMillis = millis();

  //These are interrupt functions to count encoders and ensure emergency stop conditions
  //attachInterrupt(digitalPinToInterrupt(encoderHorizPin), ISRrotChangeEncHoriz, CHANGE);
  //attachInterrupt(digitalPinToInterrupt(encoderVertPin1), ISRrotChangeVert1, CHANGE);
  //attachInterrupt(digitalPinToInterrupt(encoderVertPin2), ISRrotChangeVert2, CHANGE);
}


void loop() {
  currentMillis = millis();  //get the current "time" (actually the number of milliseconds since the program started)
  if (currentMillis - heartbeatMillis >= HEARTBEATPERIOD)  //test whether the period has elapsed
  {
    digitalWrite(led, !digitalRead(led));
    heartbeatMillis = currentMillis;  //IMPORTANT to save the start time of the current LED state.
  }

  //This runs things are a defined rate
  if (currentMillis - publishMillis >= PUBLISHPERIOD)
  {
    //this is commented out but should be used to publish joint states back to ROS
    publishStates();

    // Publish the Gripper positino back to ROS
    if (enableGripper) {
      getGripperPosition();
    }

    // Publish the Gripper positino back to ROS
    if (enablePrune) {
      getPrunePosition();
    }

    publishMillis = currentMillis; //IMPORTANT to save the start time of the current publish state.

  }

  //this is needed to update ROSSERIAL callbacks
  nh.spinOnce();//

  //This only runs motors after homing sequence is finished
  if (homingFinished) {
    //These are needed to ensure AccelStepper libraries continue running steppers to set positions
    liftAxis1.run();
    liftAxis2.run();
    horizontalAxis.run();

    //      //If current position within a certain tolerance of desired position, STOP motors
    //      if ((horizontalAxis.currentPosition() < horizontalAxis.targetPosition() + 5) && (horizontalAxis.currentPosition() > horizontalAxis.targetPosition() - 5)) {
    //        horizontalAxis.setCurrentPosition(0);
    //      } else {
    //        horizontalAxis.run();
    //      }
  }


  // I'm not sure if this code is still needed or handled in interrupt routine
  //    long lifAxis1EncoderCurrPos = positionEncoder0;
  //    if(liftAxis1.isRunning()) {
  //       if (lifAxis1EncoderCurrPos == positionEncoder0) {
  //        enc0DiffCounter++;
  //      }
  //    }

  //    long lifAxis2EncoderCurrPos = positionEncoder1;
  //    if(liftAxis2.isRunning()) {
  //      if (lifAxis2EncoderCurrPos == positionEncoder1) {
  //        enc1DiffCounter++;
  //      }
  //    }

  //  if (enc0DiffCounter >= ENCMOTTHRESH || enc1DiffCounter >= ENCMOTTHRESH) {
  //     triggerEstop();
  //     nh.loginfo("Encoder motor threshold reached. Triggering ESTOP");
  //     enc0DiffCounter = 0;
  //     enc1DiffCounter = 0;
  //   }

}


//Publisher callback for ROSSERIAL
void publishStates() {
  unsigned long readMillis1, readMillis2, readMillis3, readMillis4, readMillis5, readMillis6, timetoread, timetoread2, timetoread3;
  char log_msg1[32], log_msg2[32], log_msg3[32];


  motor_states.data[0] = STEPS_TO_METRES((float)liftAxis1.currentPosition());
  motor_states.data[1] = STEPS_TO_METRES_HORIZONTAL((float)horizontalAxis.currentPosition());

  motor_states.data[2] = 0;
  motor_states.data[3] = 0;
  motor_states.data[4] = 0;



  //  timetoread2 = readMillis4 - readMillis3;
  //  timetoread3 = readMillis6 - readMillis5;

  //  sprintf(log_msg1,"dxl read command took: %ld", timetoread);
  //  nh.loginfo(log_msg1);
  //  motor_states_pub.publish(&motor_states);

  //  sprintf(log_msg2,"dxl second read command took: %ld", timetoread2);
  //  nh.loginfo(log_msg2);
  //  motor_states_pub.publish(&motor_states);
  //
  //
  //  sprintf(log_msg3,"dxl third read command took: %ld", timetoread2);
  //  nh.loginfo(log_msg3);
  motor_states_pub.publish(&motor_states);

}

//publisher for sending homing state signal
void publishHomingStatus() {
  nh.loginfo("Publishing Homing Status");
  homing_pub.publish(&homing_state);
}


//This function is used to set current positions of the stepper motors to end stop values
//This needs testing
//uses MACROS defined at top of file
void setEndStopPosition() {

  // Reset encoder positions to initial state
  positionEncoderHoriz = METRES_TO_COUNTS(MIN_HORIZONTAL_POSITION_METRES);
  positionEncoderVert1 = METRES_TO_COUNTS(MIN_LIFT_POSITION_METRES);
  positionEncoderVert2 = METRES_TO_COUNTS(MIN_LIFT_POSITION_METRES);

  liftAxis1.setCurrentPosition(METRES_TO_STEPS(MIN_LIFT_POSITION_METRES));
  liftAxis2.setCurrentPosition(METRES_TO_STEPS(MIN_LIFT_POSITION_METRES));
  horizontalAxis.setCurrentPosition(METRES_TO_STEPS_HORIZONTAL(MIN_HORIZONTAL_POSITION_METRES));

}


//This function runs the motors until limit switches are hit and then sends a signal back to ROS saying its finishing homing behaviour
void homingSequence() {
  nh.loginfo("Horizontal homing sequence in progress");

  horizontalAxis.setSpeed(-MICRO_PER_STEP * 200); // Default 20000
  liftAxis1.setSpeed(-MICRO_PER_STEP * 100);
  liftAxis2.setSpeed(-MICRO_PER_STEP * 100);
  //
  while (digitalRead(xLimitPin) != LOW) {
    nh.spinOnce();
    horizontalAxis.runSpeed();
  }
  horizontalAxis.stop();


  nh.loginfo("Horizontal homing sequence completed");


  bool yLimitReached = false;
  bool zLimitReached = false;
  while (!yLimitReached || !zLimitReached) {
    nh.spinOnce();

    if (digitalRead(zLimitPin) == LOW && !zLimitReached) {
      nh.loginfo("zLimit Pin triggered");
      zLimitReached = true;
      liftAxis1.stop();
    } else {
      liftAxis1.runSpeed();
    }

    if (digitalRead(yLimitPin) == LOW && !yLimitReached) {
      nh.loginfo("yLimit Pin triggered");
      yLimitReached = true;
      liftAxis2.stop();
    } else {
      liftAxis2.runSpeed();
    }
  }

  // Reset encoder positions to initial state
  setEndStopPosition();

  char log_msg1[32];
  sprintf(log_msg1, "lift axis 1 current position: %ld", liftAxis1.currentPosition());
  nh.loginfo(log_msg1);

  char log_msg2[32];
  sprintf(log_msg2, "lift axis 2 current position: %ld", liftAxis2.currentPosition());
  nh.loginfo(log_msg2);

  char log_msg3[32];
  sprintf(log_msg3, "horizontal 3 current position: %ld", horizontalAxis.currentPosition());
  nh.loginfo(log_msg3);

  //
  //  while (1) {
  //      nh.spinOnce();
  //      liftAxis1.setSpeed(-5000);
  //      liftAxis2.setSpeed(-5000);
  //  }

  nh.loginfo("Vertical homing sequence completed");
}


//Interrupt functions for horizontal encoder
void ISRrotChangeEncHoriz() {
  newEncoderHorizPos = encoderHoriz.read();
  positionEncoderHoriz = newEncoderHorizPos;

}

//Interrupt functions for vertical encoders
//Includes emergency stop condition check
//If encoders are out of sync
//these were ensure newEncoderVert1Pos variable are interrupt safe variables (volatile)
void ISRrotChangeVert1() {
  newEncoderVert1Pos = encoderVert1.read();
  positionEncoderVert1 = newEncoderVert1Pos;
  if (abs(positionEncoderVert1 - positionEncoderVert2) > ENCTHRESH)
  {
    nh.loginfo("Encoder Threshold reached. E-stop Triggered");
    triggerEstop();

  }
}


void ISRrotChangeVert2() {
  newEncoderVert2Pos = encoderVert2.read();
  positionEncoderVert2 = newEncoderVert2Pos;
  if (abs(positionEncoderVert1 - positionEncoderVert2) > ENCTHRESH)
  {
    nh.loginfo("Encoder Threshold reached. E-stop Triggered");
    triggerEstop();
  }
}

//Kills motor stepper drivers
void triggerEstop() {
  digitalWrite(EN, HIGH);
}
