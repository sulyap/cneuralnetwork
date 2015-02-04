#!/usr/bin/env ruby
# Tool to normalize a data file
# Syntax: ruby normalizer.rb [input_file] [output_file] [meta_file]

require 'csv'

def syntax
  puts "normalizer.rb [input_file] [output_file] [meta_file]"
end

# Variables
num_rows = 0
num_cols = 0
min_values = []
max_values = []
data = []
scaled_data = []

# error checking
if ARGV.length != 3
  syntax
  abort
end

# read values
input_file = ARGV[0]

# Get number of rows
num_rows = CSV.read(input_file).length
puts "Number of rows: #{num_rows}"

# Get number of cols
CSV.foreach(input_file) do |row|
  num_cols = row.size
end
puts "Number of cols: #{num_cols}"

# Get raw data
CSV.foreach(input_file) do |row|
  data << row
end

scaled_data = data

# build min max values
puts "Building min max values..."
num_cols.times do |col_counter|
  min_value = 99999999999
  max_value = 0.00

  num_rows.times do |row_counter|
    if data[row_counter][col_counter].to_f < min_value
      min_value = data[row_counter][col_counter].to_f
    end

    if data[row_counter][col_counter].to_f >= max_value
      max_value = data[row_counter][col_counter].to_f
    end
  end

  min_values[col_counter] = min_value
  max_values[col_counter] = max_value
end

# scale_data(x) -> if x < 0 then x = x + |x_min|
puts "Scaling data..."
num_cols.times do |col_counter|
  num_rows.times do |row_counter|
    if data[row_counter][col_counter].to_f < 0.00
      scaled_data[row_counter][col_counter] = data[row_counter][col_counter].to_f + min_values[col_counter].abs
    else
      scaled_data[row_counter][col_counter] = data[row_counter][col_counter].to_f
    end
  end
end

# build min max values based on scaled_data
puts "Buildling new min max values..."
num_cols.times do |col_counter|
  min_value = 99999999999
  max_value = 0.00

  num_rows.times do |row_counter|
    if scaled_data[row_counter][col_counter].to_f < min_value
      min_value = scaled_data[row_counter][col_counter].to_f
    end

    if scaled_data[row_counter][col_counter].to_f >= max_value
      max_value = scaled_data[row_counter][col_counter].to_f
    end
  end

  min_values[col_counter] = min_value
  max_values[col_counter] = max_value
end


# rebuild data from scaled_data, stdev_values, mean_values
puts "Rebuilding scaled data..."
normalized_data = scaled_data
scaled_data.each_with_index do |d_array, r|
  d_array.each_with_index do |x, c|
    if min_values[c] == 0.00
      n = x / max_values[c]
    else
      n = (x - min_values[c]) / (max_values[c] - min_values[c])
    end

    normalized_data[r][c] = n
  end
end

# build output csv
puts "Creating output file #{ARGV[1]}..."
output_file = File.new(ARGV[1], "w")
normalized_data.each do |normalized_vector|
  output_file.puts normalized_vector.to_csv
end
output_file.close

# build meta file
puts "Creating metadata file #{ARGV[2]}..."
meta_file = File.new(ARGV[2], "w")
meta_file.puts "num_dimensions=#{num_cols}"
meta_file.puts "num_samples=#{num_rows}"
meta_file.puts "min_values=#{min_values}"
meta_file.puts "max_values=#{max_values}"
meta_file.close

puts "Done..."
