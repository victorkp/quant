#!/usr/bin/perl
use strict;
use warnings;

use Data::Dumper;
use DateTime;
use File::Slurp;

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
    $header = $header . $ARGV[0] . " value, " . $ARGV[0] . " chg pct,";

    my @stats;
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
            push @datapoint, $price; # push absolute price/data
            if(scalar(@stats)) {
                my $last = $stats[-1][1];
                push @datapoint, 100 * ((1.0 * $price - $last) / $last);
            } else {
                push @datapoint, 0;
            }

            push @stats, \@datapoint;
        }
    }
    push @quarter_stats, \@stats;
    shift @ARGV;
}
shift @ARGV; # Shift out 'break' keyword

for my $file (@ARGV) {
    open(IN, "<$file") or die "Couldn't open file in";

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

            # Output: low, high, close
            if(!exists($data{$dt->epoch})) {
                $data{$dt->epoch} = "$6,$5,$7";
            } else {
                $data{$dt->epoch} = $data{$dt->epoch} . ",$6,$5,$7";
            }
        } else {
            $header = $header . "$file LOW, $file HIGH, $file CLOSE,";
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
            print OUT "$d,";
        }
    }

    print OUT $data{$date} , "\n";
}

close OUT;
