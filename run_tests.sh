#!/bin/bash

pytest --exitfirst --verbose --failed-first --cov=. --cov-report html
