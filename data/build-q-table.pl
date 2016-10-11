#!/usr/bin/perl
use strict;
use warnings;

use Data::Dumper;
use DateTime;
use File::Slurp;

## Goal: at any point in time, give maximum possible future reward
# Input: <output Q table file> <output from build-market-input.pl> 

if(scalar(@ARGV) < 1) {
    die "Need input/output file argument";
}

open(OUT, ">$ARGV[0]") or die "Couldn't open file out";

my @data_points = read_file($ARGV[1]) or die "Couldn't open input file";

# What lines are values
my @close_columns;

# First line is header, find what columns are CLOSED prices
my @header = split(/,/, $data_points[0]);
for(my $i = 0; $i < scalar(@header); $i++) {
    if($header[$i] =~ /CLOSE/) {
        # print "Closing Price Header Column: \"$header[$i]\"\n";
        push(@close_columns, $i);
    }
}

my $profit_until_end = 0;

my @dates;     # All date times stored for later use
my @q_profits; # Possible reward from a time until end of samples

my @last_prices;

# Start at last datapoint and work backwards
for(my $i = scalar(@data_points) - 1; $i > 0; $i--) {
    my @data_point = split(/,/, $data_points[$i]);
    my $date = $data_point[0];
    push(@dates, $date);
    my @prices;
    for my $column (@close_columns) {
        push(@prices, $data_point[$column]);
    }

    if(scalar(@last_prices) != 0) {
        # Find best investment for time period until @last_prices
        my $biggest_profit = 0; # Worst case is we have no market stake at all
        my $best_security = "";
        for(my $j = 0; $j < scalar(@prices); $j++) {
            if($last_prices[$j] - $prices[$j] > $biggest_profit) {
                $biggest_profit = $last_prices[$j] - $prices[$j];
                $best_security = $header[$close_columns[$j]];
            }
        }
        $profit_until_end += $biggest_profit;
        push(@q_profits, $profit_until_end);
        # printf("Time: $date, Profit: %.2f, Security: $best_security, Cumulative: %.2f\n", $biggest_profit, $profit_until_end);
    }

    @last_prices = @prices;
}

# Write out Q Table
for(my $i = scalar(@dates) - 2; $i >= 0; $i--) {
    print OUT "$dates[$i],$q_profits[$i]\n";
}

close OUT;
