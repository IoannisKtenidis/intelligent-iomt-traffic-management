#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 LoRaSim 0.2.1: simulate collisions in LoRa
 Copyright © 2016 Thiemo Voigt <thiemo@sics.se> and Martin Bor <m.bor@lancaster.ac.uk>

 This work is licensed under the Creative Commons Attribution 4.0
 International License. To view a copy of this license,
 visit http://creativecommons.org/licenses/by/4.0/.

 Do LoRa Low-Power Wide-Area Networks Scale? Martin Bor, Utz Roedig, Thiemo Voigt
 and Juan Alonso, MSWiM '16, http://dx.doi.org/10.1145/2988287.2989163

 $Date: 2017-05-12 19:16:16 +0100 (Fri, 12 May 2017) $
 $Revision: 334 $
"""

"""
 SYNOPSIS:
    ./loraDir.py <nodes> <avgsend> <experiment> <simtime> [collision]
 DESCRIPTION:
    nodes
        number of nodes to simulate
    avgsend
        average sending interval in milliseconds
    experiment
        experiment is an integer that determines with what radio settings the
        simulation is run. All nodes are configured with a fixed transmit power
        and a single transmit frequency, unless stated otherwise.
        0   use the settings with the slowest datarate (SF12, BW125, CR4/8).
        1   similar to experiment 0, but use a random choice of 3 transmit
            frequencies.
        2   use the settings with the fastest data rate (SF6, BW500, CR4/5).
        3   optimise the setting per node based on the distance to the gateway.
        4   use the settings as defined in LoRaWAN (SF12, BW125, CR4/5).
        5   similar to experiment 3, but also optimises the transmit power.
    simtime
        total running time in milliseconds
    collision
        set to 1 to enable the full collision check, 0 to use a simplified check.
        With the simplified check, two messages collide when they arrive at the
        same time, on the same frequency and spreading factor. The full collision
        check considers the 'capture effect', whereby a collision of one or the
 OUTPUT
     The result of every simulation run will be appended to a file named expX.dat,
     whereby X is the experiment number. The file contains a space separated table
     of values for nodes, collisions, transmissions and total energy spent. The
     data file can be easily plotted using e.g. gnuplot.
"""

import simpy
import random
import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import os

# turn on/off graphics
graphics = 0

# do the full collision check
full_collision = False

# Scenario Selection: 1, 2, or 3
# Scenario 1: p = 0.018, q = 0.764
# Scenario 2: p = 0.020, q = 0.849
# Scenario 3: p = 0.022, q = 0.934
scenario = 1

# Number of intervals to use as features (1, 2, 3, or 4)
k_intervals = 3

# Enable/disable Listen-Before-Talk (LBT) mechanism (True for proposed, False for ALOHA baseline)
use_lbt = True


if scenario == 1:
    p_trans = 0.018
    q_trans = 0.764
elif scenario == 2:
    p_trans = 0.020
    q_trans = 0.849
elif scenario == 3:
    p_trans = 0.022
    q_trans = 0.934
else:
    print("Error: Invalid scenario. Choose 1, 2, or 3.")
    sys.exit(-1)

print(f"Selected Scenario: {scenario} (p = {p_trans}, q = {q_trans})")
print(f"Features count (K-intervals): {k_intervals}")

# Load XGBoost model
import xgboost as xgb
model_file = os.path.join(os.path.dirname(__file__), f"xgb_model_scenario{scenario}_k{k_intervals}.json")
xgb_model = xgb.XGBClassifier()
if os.path.exists(model_file):
    xgb_model.load_model(model_file)
    print(f"Loaded XGBoost model: {model_file}")
else:
    print(f"Warning: Model file {model_file} not found. Running with true-state fallback.")
    xgb_model = None



# experiments:
# 0: packet with longest airtime, aloha-style experiment
# 0: one with 3 frequencies, 1 with 1 frequency
# 2: with shortest packets, still aloha-style
# 3: with shortest possible packets depending on distance


# this is an array with measured values for sensitivity
# see paper, Table 3
sf7 = np.array([7,-126.5,-124.25,-120.75])
sf8 = np.array([8,-127.25,-126.75,-124.0])
sf9 = np.array([9,-131.25,-128.25,-127.5])
sf10 = np.array([10,-132.75,-130.25,-128.75])
sf11 = np.array([11,-134.5,-132.75,-128.75])
sf12 = np.array([12,-133.25,-132.25,-132.25])

# Note: called before a packet (or rather node) is inserted into the list
def checkcollision(packet):
    """
    Checks if the given packet collides with any other packets currently being
    received at the base station (BS).
    
    Parameters:
        packet (myPacket): The packet to check for collisions.
        
    Returns:
        int: 1 if a collision occurred that ruins this packet, 0 otherwise.
    """
    col = 0 # flag needed since there might be several collisions for packet
    processing = 0
    for i in range(0,len(packetsAtBS)):
        if packetsAtBS[i].packet.processed == 1:
            processing = processing + 1
            
    # Gateway capacity constraint: if concurrent receptions exceed maxBSReceives,
    # the gateway cannot process the new packet
    if (processing > maxBSReceives):
        print("too long:", len(packetsAtBS))
        packet.processed = 0
    else:
        packet.processed = 1

    if packetsAtBS:
        print("CHECK node {} (sf:{} bw:{} freq:{:.6e}) others: {}".format(
             packet.nodeid, packet.sf, packet.bw, packet.freq,
             len(packetsAtBS)))
        for other in packetsAtBS:
            if other.nodeid != packet.nodeid:
               print(">> node {} (sf:{} bw:{} freq:{:.6e})".format(
                   other.nodeid, other.packet.sf, other.packet.bw, other.packet.freq))
               
               # Check for overlap in frequency AND spreading factor
               if frequencyCollision(packet, other.packet) \
                   and sfCollision(packet, other.packet):
                   if full_collision:
                       # Full collision model checks timing and power (capture effect)
                       if timingCollision(packet, other.packet):
                           c = powerCollision(packet, other.packet)
                           for p in c:
                               p.collided = 1
                               if p == packet:
                                   col = 1
                       else:
                           pass
                   else:
                       # Simplified collision model: overlapping packets on same SF/freq always collide
                       packet.collided = 1
                       other.packet.collided = 1  # other also got lost
                       col = 1
        return col
    return 0

def frequencyCollision(p1, p2):
    """
    Determines if two packets overlap in frequency based on their bandwidths.
    
    Collision conditions:
        - |f1 - f2| <= 120 kHz if either has BW = 500 kHz
        - |f1 - f2| <= 60 kHz if either has BW = 250 kHz
        - |f1 - f2| <= 30 kHz if either has BW = 125 kHz
    """
    if (abs(p1.freq-p2.freq)<=120000 and (p1.bw==500 or p2.bw==500)):
        print("frequency coll 500")
        return True
    elif (abs(p1.freq-p2.freq)<=60000 and (p1.bw==250 or p2.bw==250)):
        print("frequency coll 250")
        return True
    else:
        if (abs(p1.freq-p2.freq)<=30000):
            print("frequency coll 125")
            return True
    print("no frequency coll")
    return False

def sfCollision(p1, p2):
    """
    Determines if two packets share the same Spreading Factor (SF).
    LoRa channels are orthogonal across different SFs.
    """
    if p1.sf == p2.sf:
        print("collision sf node {} and node {}".format(p1.nodeid, p2.nodeid))
        return True
    print("no sf collision")
    return False

def powerCollision(p1, p2):
    """
    Evaluates the Capture Effect in LoRa: if one packet is significantly
    stronger than the other (>= 6 dB), it is received successfully (survives).
    
    Returns:
        tuple: Packets that collided/failed.
    """
    powerThreshold = 6 # dB
    print("pwr: node {0.nodeid} {0.rssi:3.2f} dBm node {1.nodeid} {1.rssi:3.2f} dBm; diff {2:3.2f} dBm".format(p1, p2, round(p1.rssi - p2.rssi,2)))
    if abs(p1.rssi - p2.rssi) < powerThreshold:
        # Both packets are within 6 dB, so both collide
        print("collision pwr both node {} and node {}".format(p1.nodeid, p2.nodeid))
        return (p1, p2)
    elif p1.rssi - p2.rssi < powerThreshold:
        # p2 overpowered p1, so p1 is lost
        print("collision pwr node {} overpowered node {}".format(p2.nodeid, p1.nodeid))
        return (p1,)
    # p1 overpowered p2, so p2 is lost
    print("p1 wins, p2 lost")
    return (p2,)

def timingCollision(p1, p2):
    """
    Determines if packet p1 arrived late enough that it does not disrupt p2's
    preamble, or vice versa.
    """
    Npream = 8
    # Calculate preamble duration in ms
    Tpreamb = 2**p1.sf/(1.0*p1.bw) * (Npream - 5)
    p2_end = p2.addTime + p2.rectime
    p1_cs = env.now + Tpreamb
    print("collision timing node {} ({},{},{}) node {} ({},{})".format(
        p1.nodeid, env.now - env.now, p1_cs - env.now, p1.rectime,
        p2.nodeid, p2.addTime - env.now, p2_end - env.now
    ))
    if p1_cs < p2_end:
        print("not late enough")
        return True
    print("saved by the preamble")
    return False

def airtime(sf, cr, pl, bw):
    """
    Calculates the Time-on-Air (ToA) of a LoRa packet in milliseconds.
    Formula derived from the LoRa Design Guide.
    
    Parameters:
        sf (int): Spreading Factor (6 to 12)
        cr (int): Coding Rate index (1 to 4)
        pl (int): Payload size in bytes
        bw (int): Bandwidth in kHz (125, 250, 500)
    """
    H = 0        # implicit header disabled (H=0) or not (H=1)
    DE = 0       # low data rate optimization enabled (=1) or not (=0)
    Npream = 8   # number of preamble symbol (12.25  from Utz paper)

    # Low data rate optimization is mandatory for SF11 & SF12 with BW125
    if bw == 125 and sf in [11, 12]:
        DE = 1
    # Implicit header is required for SF6
    if sf == 6:
        H = 1

    Tsym = (2.0**sf)/bw
    Tpream = (Npream + 4.25)*Tsym
    print("sf", sf, " cr", cr, "pl", pl, "bw", bw)
    payloadSymbNB = 8 + max(math.ceil((8.0*pl-4.0*sf+28+16-20*H)/(4.0*(sf-2*DE)))*(cr+4),0)
    Tpayload = payloadSymbNB * Tsym
    return Tpream + Tpayload

#
# this function creates a node
#
class myNode():
    def __init__(self, nodeid, bs, period, packetlen):
        self.nodeid = nodeid
        self.period = period
        self.bs = bs
        self.x = 0
        self.y = 0

        # this is very complex procedure for placing nodes
        # and ensure minimum distance between each pair of nodes
        found = 0
        rounds = 0
        global nodes
        while (found == 0 and rounds < 100):
            a = random.random()
            b = random.random()
            if b<a:
                a,b = b,a
            posx = b*maxDist*math.cos(2*math.pi*a/b)+bsx
            posy = b*maxDist*math.sin(2*math.pi*a/b)+bsy
            if len(nodes) > 0:
                for index, n in enumerate(nodes):
                    dist = np.sqrt(((abs(n.x-posx))**2)+((abs(n.y-posy))**2))
                    if dist >= 10:
                        found = 1
                        self.x = posx
                        self.y = posy
                    else:
                        rounds = rounds + 1
                        if rounds == 100:
                            print("could not place new node, giving up")
                            exit(-1)
            else:
                print("first node")
                self.x = posx
                self.y = posy
                found = 1
        self.dist = np.sqrt((self.x-bsx)*(self.x-bsx)+(self.y-bsy)*(self.y-bsy))
        print('node %d' %nodeid, "x", self.x, "y", self.y, "dist: ", self.dist)

        self.packet = myPacket(self.nodeid, packetlen, self.dist)
        self.sent = 0

        # IoMT Markov state: Healthy (Routine) or Not-Healthy (Critical)
        # Initialized based on stationary probabilities: q / (p + q)
        self.state = 'Healthy' if random.random() < (q_trans / (p_trans + q_trans)) else 'Not-Healthy'
        
        # History of recent transmission intervals (in ms)
        # Initialize with representative intervals for the initial state
        self.interval_history = []
        for _ in range(k_intervals):
            if self.state == 'Healthy':
                lam = 1.96
            else:
                lam = 2.72 if random.random() < 0.257 else 12.0
            self.interval_history.append(random.expovariate(lam / 86400000.0))
        
        # Predicted state using the model or fallback
        self.predicted_state = self.classify_state()

    def classify_state(self):
        """
        Classifies the current state of the node (Healthy vs Not-Healthy) using 
        the loaded XGBoost model based on the recent interval history.
        If no model is available, falls back to the true state.
        """
        global xgb_model
        if xgb_model is not None:
            # Reverse interval history to match training feature order (newest first, i.e., dt_0, dt_1, ...)
            reversed_history = self.interval_history[::-1]
            features = np.array([reversed_history]).reshape(1, -1)
            pred = xgb_model.predict(features)[0]
            return 'Not-Healthy' if pred == 1 else 'Healthy'
        else:
            # Fallback to true state if no model loaded
            return self.state




        # graphics for node
        global graphics
        if (graphics == 1):
            global ax
            ax.add_artist(plt.Circle((self.x, self.y), 2, fill=True, color='blue'))

#
# this function creates a packet (associated with a node)
# it also sets all parameters, currently random
#
class myPacket():
    def __init__(self, nodeid, plen, distance):
        global experiment
        global Ptx
        global gamma
        global d0
        global var
        global Lpld0
        global GL

        self.nodeid = nodeid
        self.txpow = Ptx

        # randomize configuration values
        self.sf = random.randint(6,12)
        self.cr = random.randint(1,4)
        self.bw = random.choice([125, 250, 500])

        # for certain experiments override these
        if experiment==1 or experiment == 0:
            self.sf = 12
            self.cr = 4
            self.bw = 125

        # for certain experiments override these
        if experiment==2:
            self.sf = 6
            self.cr = 1
            self.bw = 500
        # lorawan
        if experiment == 4:
            self.sf = 12
            self.cr = 1
            self.bw = 125


        # for experiment 3 find the best setting
        # OBS, some hardcoded values
        Prx = self.txpow  ## zero path loss by default

        # log-shadow
        Lpl = Lpld0 + 10*gamma*math.log10(distance/d0)
        print("Lpl:", Lpl)
        Prx = self.txpow - GL - Lpl

        if (experiment == 3) or (experiment == 5):
            minairtime = 9999
            minsf = 0
            minbw = 0

            print("Prx:", Prx)

            for i in range(0,6):
                for j in range(1,4):
                    if (sensi[i,j] < Prx):
                        self.sf = int(sensi[i,0])
                        if j==1:
                            self.bw = 125
                        elif j==2:
                            self.bw = 250
                        else:
                            self.bw=500
                        at = airtime(self.sf, 1, plen, self.bw)
                        if at < minairtime:
                           minairtime = at
                           minsf = self.sf
                           minbw = self.bw
                           minsensi = sensi[i, j]
            if (minairtime == 9999):
                print("does not reach base station")
                exit(-1)
            print("best sf:", minsf, " best bw: ", minbw, "best airtime:", minairtime)
            self.rectime = minairtime
            self.sf = minsf
            self.bw = minbw
            self.cr = 1

            if experiment == 5:
                # reduce the txpower if there's room left
                self.txpow = max(2, self.txpow - math.floor(Prx - minsensi))
                Prx = self.txpow - GL - Lpl
                print('minsesi {} best txpow {}'.format(minsensi, self.txpow))

        # transmission range, needs update XXX
        self.transRange = 150
        self.pl = plen
        self.symTime = (2.0**self.sf)/self.bw
        self.arriveTime = 0
        self.rssi = Prx
        # frequencies: lower bound + number of 61 Hz steps
        self.freq = 860000000 + random.randint(0,2622950)

        # for certain experiments override these and
        # choose some random frequences
        if experiment == 1:
            self.freq = random.choice([860000000, 864000000, 868000000])
        else:
            self.freq = 860000000

        print("frequency", self.freq, "symTime ", self.symTime)
        print("bw", self.bw, "sf", self.sf, "cr", self.cr, "rssi", self.rssi)
        self.rectime = airtime(self.sf,self.cr,self.pl,self.bw)
        print("rectime node ", self.nodeid, "  ", self.rectime)
        # denote if packet is collided
        self.collided = 0
        self.processed = 0

#
# main discrete event loop, runs for each node
# a global list of packet being processed at the gateway
# is maintained
#
def transmit(env,node):
    while True:
        # 1. State transition
        if node.state == 'Healthy':
            if random.random() < p_trans:
                node.state = 'Not-Healthy'
        else: # node.state == 'Not-Healthy'
            if random.random() < q_trans:
                node.state = 'Healthy'

        # 2. Get lambda based on current predicted state (from the last event)
        if node.predicted_state == 'Healthy':
            lam = 1.96
        else:
            # Mixture of two Poisson distributions (74.3% chronic with rate 12.0, 25.7% acute with rate 2.72)
            if random.random() < 0.257:
                lam = 2.72
            else:
                lam = 12.0

        # Calculate transmission interval in ms (units of daily measurements)
        interval = random.expovariate(lam / 86400000.0)
        yield env.timeout(interval)

        # After waking up, update history with this interval and classify state
        node.interval_history.append(interval)
        if len(node.interval_history) > k_intervals:
            node.interval_history.pop(0)
            
        node.predicted_state = node.classify_state()

        # Track classifier stats
        global tp_class, tn_class, fp_class, fn_class
        if node.state == 'Not-Healthy' and node.predicted_state == 'Not-Healthy':
            tp_class += 1
        elif node.state == 'Healthy' and node.predicted_state == 'Healthy':
            tn_class += 1
        elif node.state == 'Healthy' and node.predicted_state == 'Not-Healthy':
            fp_class += 1
        elif node.state == 'Not-Healthy' and node.predicted_state == 'Healthy':
            fn_class += 1

        # 3. LBT Mechanism (only for predicted Healthy nodes if LBT is enabled)
        if use_lbt and node.predicted_state == 'Healthy':

            while True:
                channel_busy = False
                for other in packetsAtBS:
                    # check if channel is occupied by predicted Not-Healthy traffic on colliding frequency
                    if other.predicted_state == 'Not-Healthy' and frequencyCollision(node.packet, other.packet):
                        channel_busy = True
                        break
                if not channel_busy:
                    break
                else:
                    global nrDeferred
                    nrDeferred += 1
                    # Defer transmission: wait for random backoff time (1 to 5 seconds)
                    yield env.timeout(random.uniform(1000, 5000))



        # time sending and receiving
        # packet arrives -> add to base station

        node.sent = node.sent + 1
        if (node in packetsAtBS):
            print("ERROR: packet already in")
        else:
            sensitivity = sensi[node.packet.sf - 7, [125,250,500].index(node.packet.bw) + 1]
            if node.packet.rssi < sensitivity:
                print("node {}: packet will be lost".format(node.nodeid))
                node.packet.lost = True
            else:
                node.packet.lost = False
                # adding packet if no collision
                if (checkcollision(node.packet)==1):
                    node.packet.collided = 1
                else:
                    node.packet.collided = 0
                packetsAtBS.append(node)
                node.packet.addTime = env.now

        yield env.timeout(node.packet.rectime)

        if node.packet.lost:
            global nrLost
            nrLost += 1
        if node.packet.collided == 1:
            global nrCollisions
            nrCollisions = nrCollisions + 1
        if node.packet.collided == 0 and not node.packet.lost:
            global nrReceived
            nrReceived = nrReceived + 1
        if node.packet.processed == 1:
            global nrProcessed
            nrProcessed = nrProcessed + 1

        # complete packet has been received by base station
        # can remove it
        if (node in packetsAtBS):
            packetsAtBS.remove(node)
            # reset the packet
        node.packet.collided = 0
        node.packet.processed = 0
        node.packet.lost = False

#
# "main" program
#

# get arguments
if len(sys.argv) >= 5:
    nrNodes = int(sys.argv[1])
    avgSendTime = int(sys.argv[2])
    experiment = int(sys.argv[3])
    simtime = int(sys.argv[4])
    if len(sys.argv) > 5:
        full_collision = bool(int(sys.argv[5]))
    if len(sys.argv) > 6:
        scenario = int(sys.argv[6])
        if scenario == 1:
            p_trans = 0.018
            q_trans = 0.764
        elif scenario == 2:
            p_trans = 0.020
            q_trans = 0.849
        elif scenario == 3:
            p_trans = 0.022
            q_trans = 0.934
        else:
            print("Error: Invalid scenario. Choose 1, 2, or 3.")
            sys.exit(-1)
    if len(sys.argv) > 7:
        k_intervals = int(sys.argv[7])
    if len(sys.argv) > 8:
        use_lbt = bool(int(sys.argv[8]))
    print("Nodes:", nrNodes)
    print("AvgSendTime (exp. distributed):", avgSendTime)
    print("Experiment: ", experiment)
    print("Simtime: ", simtime)
    print("Full Collision: ", full_collision)
    print(f"Active Scenario: {scenario} (p = {p_trans}, q = {q_trans})")
    print(f"Features count (K-intervals): {k_intervals}")
    print(f"LBT Enabled: {use_lbt}")

    # Reload model
    model_file = os.path.join(os.path.dirname(__file__), f"xgb_model_scenario{scenario}_k{k_intervals}.json")
    xgb_model = xgb.XGBClassifier()
    if os.path.exists(model_file):
        xgb_model.load_model(model_file)
        print(f"Loaded XGBoost model: {model_file}")
    else:
        print(f"Warning: Model file {model_file} not found. Running with true-state fallback.")
        xgb_model = None
else:
    print("usage: ./loraDir <nodes> <avgsend> <experiment> <simtime> [collision] [scenario] [k_intervals] [use_lbt]")


    print("experiment 0 and 1 use 1 frequency only")
    exit(-1)



# global stuff
#Rnd = random.seed(12345)
nodes = []
packetsAtBS = []
env = simpy.Environment()

# maximum number of packets the BS can receive at the same time
maxBSReceives = 8


# max distance: 300m in city, 3000 m outside (5 km Utz experiment)
# also more unit-disc like according to Utz
bsId = 1
nrCollisions = 0
nrReceived = 0
nrProcessed = 0
nrLost = 0
nrDeferred = 0

# Classifier statistics
tp_class = 0
tn_class = 0
fp_class = 0
fn_class = 0


Ptx = 14
gamma = 2.08
d0 = 40.0
var = 0           # variance ignored for now
Lpld0 = 127.41
GL = 0

sensi = np.array([sf7,sf8,sf9,sf10,sf11,sf12])
if experiment in [0,1,4]:
    minsensi = sensi[5,2]  # 5th row is SF12, 2nd column is BW125
elif experiment == 2:
    minsensi = -112.0   # no experiments, so value from datasheet
elif experiment in [3,5]:
    minsensi = np.amin(sensi) ## Experiment 3 can use any setting, so take minimum
Lpl = Ptx - minsensi
print("amin", minsensi, "Lpl", Lpl)
maxDist = d0*(math.e**((Lpl-Lpld0)/(10.0*gamma)))
print("maxDist:", maxDist)

# base station placement
bsx = maxDist+10
bsy = maxDist+10
xmax = bsx + maxDist + 20
ymax = bsy + maxDist + 20

# prepare graphics and add sink
if (graphics == 1):
    plt.ion()
    plt.figure()
    ax = plt.gcf().gca()
    # XXX should be base station position
    ax.add_artist(plt.Circle((bsx, bsy), 3, fill=True, color='green'))
    ax.add_artist(plt.Circle((bsx, bsy), maxDist, fill=False, color='green'))


for i in range(0,nrNodes):
    # myNode takes period (in ms), base station id packetlen (in Bytes)
    # 1000000 = 16 min
    node = myNode(i,bsId, avgSendTime,20)
    nodes.append(node)
    env.process(transmit(env,node))

#prepare show
if (graphics == 1):
    plt.xlim([0, xmax])
    plt.ylim([0, ymax])
    plt.draw()
    plt.show()

# start simulation
env.run(until=simtime)

# print stats and save into file
print("nrCollisions ", nrCollisions)

# compute energy
# Transmit consumption in mA from -2 to +17 dBm
TX = [22, 22, 22, 23,                                      # RFO/PA0: -2..1
      24, 24, 24, 25, 25, 25, 25, 26, 31, 32, 34, 35, 44,  # PA_BOOST/PA1: 2..14
      82, 85, 90,                                          # PA_BOOST/PA1: 15..17
      105, 115, 125]                                       # PA_BOOST/PA1+PA2: 18..20
# mA = 90    # current draw for TX = 17 dBm
V = 3.0     # voltage XXX
sent = sum(n.sent for n in nodes)
energy = sum(node.packet.rectime * TX[int(node.packet.txpow)+2] * V * node.sent for node in nodes) / 1e6

print("energy (in J): ", energy)
print("sent packets: ", sent)
print("collisions: ", nrCollisions)
print("received packets: ", nrReceived)
print("processed packets: ", nrProcessed)
print("lost packets: ", nrLost)
print("deferred packets: ", nrDeferred)

# data extraction rate
der = (sent-nrCollisions)/float(sent) if sent > 0 else 0
print("DER:", der)
der = (nrReceived)/float(sent) if sent > 0 else 0
print("DER method 2:", der)

# Compute classifier metrics
total_class = tp_class + tn_class + fp_class + fn_class
sensitivity = tp_class / (tp_class + fn_class) if (tp_class + fn_class) > 0 else 0
specificity = tn_class / (tn_class + fp_class) if (tn_class + fp_class) > 0 else 0
accuracy = (tp_class + tn_class) / total_class if total_class > 0 else 0

print("--- Classifier Performance ---")
print(f"TP: {tp_class}, TN: {tn_class}, FP: {fp_class}, FN: {fn_class}")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print("------------------------------")


# this can be done to keep graphics visible
if (graphics == 1):
    input('Press Enter to continue ...')

# save experiment data into a dat file that can be read by e.g. gnuplot
# name of file would be:  exp0.dat for experiment 0
fname = os.path.join(os.path.dirname(__file__), "..", "Results_Data", "exp" + str(experiment) + ".dat")
print(fname)
if os.path.isfile(fname):
    res = "\n" + str(nrNodes) + " " + str(nrCollisions) + " "  + str(sent) + " " + str(energy)
else:
    res = "#nrNodes nrCollisions nrTransmissions OverallEnergy\n" + str(nrNodes) + " " + str(nrCollisions) + " "  + str(sent) + " " + str(energy)
with open(fname, "a") as myfile:
    myfile.write(res)
myfile.close()

# with open('nodes.txt','w') as nfile:
#     for n in nodes:
#         nfile.write("{} {} {}\n".format(n.x, n.y, n.nodeid))
# with open('basestation.txt', 'w') as bfile:
#     bfile.write("{} {} {}\n".format(bsx, bsy, 0))
