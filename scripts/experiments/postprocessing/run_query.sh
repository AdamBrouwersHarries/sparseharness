#!/bin/sh

queryfile=$1
password=$2

mysql -B -us1467120 -p$password -h phantom.inf.ed.ac.uk s1467120 < $queryfile