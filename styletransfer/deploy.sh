#!/bin/bash

CODE=/home/raab/styletransfer
TARGET = /var/www/style.fiw.fhws.de/
sudo cp $Code/* -r  && cd $Target &&  sudo service apache2 restart 