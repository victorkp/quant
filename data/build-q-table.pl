#!/usr/bin/perl
use strict;
use warnings;

use Data::Dumper;
use DateTime;
use File::Slurp;
use POSIX qw( strftime );

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
    print $header[$i], "\n";
    if($header[$i] =~ /change/ && $header[$i] =~ /spy|slv|gld|uso/) {
        print "Closing Price Header Column: \"$header[$i]\"\n";
        push(@close_columns, $i);
    }
}

my $profit_until_end = 0;
my $sub_profit_until_end = 0;

my $sub_ratio = 0.12;
my $prev_sub_profit = 0.00;

my @dates;           # All date times stored for later use
my @q_profits;       # Possible reward from a time until end of samples
my @q_sub_profits;   # Sub-reward from a time until end of samples
my @transaction;     # Transaction that made profit possible

# Start at last datapoint and work backwards
for(my $i = scalar(@data_points) - 1; $i > 0; $i--) {
    my @data_point = split(/,/, $data_points[$i]);
    my $date = $data_point[0];
    push(@dates, $date);
    my @prices;
    for my $column (@close_columns) {
        push(@prices, $data_point[$column]);
    }

    my $biggest_profit = 0; # Worst case is we have no market stake at all
    my $best_security = "";
    my $best_transaction = "";
    for(my $j = 0; $j < scalar(@prices); $j++) {
        # Profit in terms of percent gain/loss
        my $profit = $prices[$j];
        if($profit > $biggest_profit) {
            $biggest_profit = $profit;
            $best_security = $header[$close_columns[$j]];
            $best_transaction = "$best_security  $profit";
        }
        $sub_profit_until_end += (1 - $sub_ratio) * $profit / scalar(@prices);
    }
    $profit_until_end += $biggest_profit;

    $sub_profit_until_end += $sub_ratio * $biggest_profit;

    # if($sub_profit_until_end < $prev_sub_profit) {
    #     $sub_profit_until_end = $prev_sub_profit;
    # } else {
    $prev_sub_profit = $sub_profit_until_end;
    # }

    push(@q_profits, $profit_until_end);
    push(@q_sub_profits, $sub_profit_until_end);
    push(@transaction, $best_transaction);
    # printf("Time: $date, Profit: %.2f, Security: $best_security, Cumulative: %.2f\n", $biggest_profit, $profit_until_end);
}

# Write out Q Table
for(my $i = scalar(@dates) - 2; $i >= 0; $i--) {
    my $formatted_date = strftime("%Y-%m-%d", localtime($dates[$i]));
    print OUT "$dates[$i],$formatted_date,$q_profits[$i],$q_sub_profits[$i],$transaction[$i]\n";
}

close OUT;
