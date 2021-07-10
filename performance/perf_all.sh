#!/bin/bash
Help()
{
   # Display Help
   echo "Run all performance scripts."
   echo
   echo "Syntax: scriptTemplate [-quick|h]"
   echo "options:"
   echo "q Run tests only on smallest graph."
   echo "h     Print this Help."
   echo
}

quick=false
while getopts ":hq" option; do
   case $option in
      h) # display Help
         Help
         exit;;
      q) # only small g
         quick=true;;
      \?) # Invalid option
         echo "Error: Invalid option"
         exit;;
   esac
done

if $quick; then
    python performance/fitness_model.py &> logs/fitness.txt --quick -tol=1e-5 -xtol=1e-6
    python performance/stripe_model_multi.py &> logs/stripe_multi.txt --quick -tol=1e-5 -xtol=1e-6
    python performance/stripe_model_inv.py &> logs/stripe_inv.txt --quick -tol=1e-5 -xtol=1e-6
    python performance/stripe_model_single.py &> logs/stripe_single.txt --quick -tol=1e-5 -xtol=1e-6
else
    python performance/fitness_model.py &> logs/fitness.txt -tol=1e-5 -xtol=1e-6
    python performance/stripe_model_multi.py &> logs/stripe_multi.txt -tol=1e-5 -xtol=1e-6
    python performance/stripe_model_inv.py &> logs/stripe_inv.txt -tol=1e-5 -xtol=1e-6
    python performance/stripe_model_single.py &> logs/stripe_single.txt -tol=1e-5 -xtol=1e-6
fi
