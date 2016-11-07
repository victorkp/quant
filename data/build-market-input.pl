#!/usr/bin/perl
use strict;
use warnings;

use Data::Dumper;
use DateTime;
use File::Slurp;
use File::Basename;

# Input: <quarterly statistics with (date,value)> break <files from yahoo>

if(scalar(@ARGV) < 2) {
    die "Need input/output file argument";
}

open(OUT, ">$ARGV[0]") or die "Couldn't open file out";
shift @ARGV;

my $header = "date,";
my %data;
my @quarter_stats;

while($ARGV[0] ne "break") {
    open(IN, "<$ARGV[0]") or die "Couldn't open file in";
    $header = $header . "$ARGV[0] % of high, $ARGV[0] chg pct,";

    my @stats;
    my $last = -1;
    my $highest = -1;
    while(<IN>) {
        chomp();
        my $line = $_;

        # Date (year-month-day), Open, High, Low, Close, Volume, Adj. Close
        if($line =~ /^(\d\d\d\d)-(\d\d)-(\d\d),([\d\.]+)$/) {
            # Combine year into time since epoch
            my $dt = DateTime->new(year   => $1,
                                   month  => $2,
                                   day    => $3,
                                   time_zone => 'America/New_York' );

            my $price = $4;

            # Output: date, price, change since last data point
            my @datapoint;
            push @datapoint, $dt->epoch();

            if($highest < $price) {
                $highest = $price;
                print "Highest is now $highest\n";
            }

            my $percent_of_highest = $price / $highest;
            push @datapoint, $percent_of_highest; # push relative price/data to high

            if($last == -1) {
                push @datapoint, 0;
            } else {
                push @datapoint, ((1.0 * $price - $last) / $last);
            }

            $last = $price;

            push @stats, \@datapoint;
        }
    }
    push @quarter_stats, \@stats;
    shift @ARGV;
}
shift @ARGV; # Shift out 'break' keyword

for my $file (@ARGV) {
    open(IN, "<$file") or die "Couldn't open file in";

    $file = basename($file, ".csv");

    my $sample_count = 0;
    my $highest_price = 0;
    my $last_price = 0;
    my $average_gain_3_day = 0;
    my $average_loss_3_day = 0;
    my $average_gain_10_day = 0;
    my $average_loss_10_day = 0;
    my $average_gain_14_day = 0;
    my $average_loss_14_day = 0;
    my $average_10_day = 0;
    my $average_20_day = 0;
    my $average_40_day = 0;
    my $average_80_day = 0;

    while(<IN>) {
        chomp();
        my $line = $_;

        # Date (year-month-day), Open, High, Low, Close, Volume, Adj. Close
        if($line =~ /^(\d\d\d\d)-(\d\d)-(\d\d),([\d\.]+),([\d\.]+),([\d\.]+),([\d\.]+),([\d\.]+),([\d\.]+)$/) {
            # Combine year into time since epoch
            my $dt = DateTime->new(year   => $1,
                                   month  => $2,
                                   day    => $3,
                                   time_zone => 'America/New_York' );

            # Output: average of (low, high, close) prices
            my $price = ($6 + $5 + $7) / 3.0;
            my $price_change = ($sample_count > 0) ? (($price - $last_price) / $price) : 0.0;

            if($price_change >= 0) {
                $average_gain_3_day =  ((2 *  $average_gain_3_day)  + $price_change) / 3.0;
                $average_gain_10_day = ((9 *  $average_gain_10_day) + $price_change) / 10.0;
                $average_gain_14_day = ((12 * $average_gain_14_day) + $price_change) / 14.0;
                $average_loss_3_day =  ((2 *  $average_loss_3_day) ) / 3.0;
                $average_loss_10_day = ((9 *  $average_loss_10_day)) / 10.0;
                $average_loss_14_day = ((12 * $average_loss_14_day)) / 14.0;
            } else {
                $average_gain_3_day =  ((2 *  $average_gain_3_day))  / 3.0;
                $average_gain_10_day = ((9 *  $average_gain_10_day)) / 10.0;
                $average_gain_14_day = ((12 * $average_gain_14_day)) / 14.0;
                $average_loss_3_day =  ((2 *  $average_loss_3_day)  - $price_change) / 3.0;
                $average_loss_10_day = ((9 *  $average_loss_10_day) - $price_change) / 10.0;
                $average_loss_14_day = ((12 * $average_loss_14_day) - $price_change) / 14.0;
            }

            my $rsi_3_day  = ($average_loss_3_day == 0) ? (100) : (100 - 100.0/(1 + $average_gain_3_day/$average_loss_3_day));
            my $rsi_10_day  = ($average_loss_10_day == 0) ? (100) : (100 - 100.0/(1 + $average_gain_10_day/$average_loss_10_day));
            my $rsi_14_day  = ($average_loss_14_day == 0) ? (100) : (100 - 100.0/(1 + $average_gain_14_day/$average_loss_14_day));

            if($sample_count == 0) {
                # Init averages to first price
                $average_10_day = $price * 1.0;
                $average_20_day = $price * 1.0;
                $average_40_day = $price * 1.0;
                $average_80_day = $price * 1.0;
            } else {
                # Keep moving averages updated
                $average_10_day = ($price + ($average_10_day *  9)) / 10.0;
                $average_20_day = ($price + ($average_20_day * 19)) / 20.0;
                $average_40_day = ($price + ($average_40_day * 39)) / 40.0;
                $average_80_day = ($price + ($average_80_day * 79)) / 80.0;
            }

            # Add data if there's enough samples to have all averages initialized
            if($sample_count > 80) {
                my $change_10_day = (($price - $average_10_day) / $average_10_day);
                my $change_20_day = (($price - $average_20_day) / $average_20_day);
                my $change_40_day = (($price - $average_40_day) / $average_40_day);
                my $change_80_day = (($price - $average_80_day) / $average_80_day);

                if(!exists($data{$dt->epoch})) {
                    $data{$dt->epoch} = "$price_change, $rsi_3_day, $rsi_10_day, $rsi_14_day, $change_10_day, $change_20_day, $change_40_day, $change_80_day ";
                } else {
                    $data{$dt->epoch} = $data{$dt->epoch} . ",$price_change, $rsi_3_day, $rsi_10_day, $rsi_14_day, $change_10_day, $change_20_day, $change_40_day, $change_80_day ";
                }
            }

            $sample_count++;
            $last_price = $price;

        } else {
            $header = $header . "$file change, $file RSI 3 day, $file RSI 10 day, $file RSI 14 day, $file average_10_day, $file average_20_day, $file average_40_day, $file average_80_day,";
        }
    }
}

print OUT "$header\n";

# print Dumper(@quarter_stats);

my @dates = sort(keys(%data));
foreach my $date (@dates) {
    print OUT "$date,";

    # Find latest available economy stats before this date
    for(my $s = 0; $s < scalar(@quarter_stats); $s++) {
        my @stat = @{$quarter_stats[$s]};

        # Starting at index, look for first date after $date
        my @data;
        my $i = 0;
        for(my $time = ${$stat[$i]}[0]; $time < $date; $time = ${$stat[$i++]}[0]) {
            @data = @{$stat[$i]};
            if($i == scalar(@stat) - 1) {
                last;
            }
        }

        if(! @data) {
            @data = @{$stat[-1]};
        }

        shift @data;
        foreach my $d (@data) {
            print OUT "$d, ";
        }
    }

    print OUT $data{$date} , "\n";
}

close OUT;
