#!/bin/bash

sudo swapoff -a
stress --vm 1 --vm-bytes 90% --vm-hang 0