module mnist_lut (
  input logic [9:0] address,        // Counter ranging from 0 to 783
  output logic [7:0] pixel       // pixel 
);

  always_comb begin
    case (address)
	  10'b0: pixel = 10'b0;
10'b1: pixel = 10'b0;
10'b10: pixel = 10'b0;
10'b11: pixel = 10'b0;
10'b100: pixel = 10'b0;
10'b101: pixel = 10'b0;
10'b110: pixel = 10'b0;
10'b111: pixel = 10'b0;
10'b1000: pixel = 10'b0;
10'b1001: pixel = 10'b0;
10'b1010: pixel = 10'b0;
10'b1011: pixel = 10'b0;
10'b1100: pixel = 10'b0;
10'b1101: pixel = 10'b0;
10'b1110: pixel = 10'b0;
10'b1111: pixel = 10'b0;
10'b10000: pixel = 10'b0;
10'b10001: pixel = 10'b0;
10'b10010: pixel = 10'b0;
10'b10011: pixel = 10'b0;
10'b10100: pixel = 10'b0;
10'b10101: pixel = 10'b0;
10'b10110: pixel = 10'b0;
10'b10111: pixel = 10'b0;
10'b11000: pixel = 10'b0;
10'b11001: pixel = 10'b0;
10'b11010: pixel = 10'b0;
10'b11011: pixel = 10'b0;
10'b11100: pixel = 10'b0;
10'b11101: pixel = 10'b0;
10'b11110: pixel = 10'b0;
10'b11111: pixel = 10'b0;
10'b100000: pixel = 10'b0;
10'b100001: pixel = 10'b0;
10'b100010: pixel = 10'b0;
10'b100011: pixel = 10'b0;
10'b100100: pixel = 10'b0;
10'b100101: pixel = 10'b0;
10'b100110: pixel = 10'b0;
10'b100111: pixel = 10'b0;
10'b101000: pixel = 10'b0;
10'b101001: pixel = 10'b0;
10'b101010: pixel = 10'b0;
10'b101011: pixel = 10'b0;
10'b101100: pixel = 10'b0;
10'b101101: pixel = 10'b0;
10'b101110: pixel = 10'b0;
10'b101111: pixel = 10'b0;
10'b110000: pixel = 10'b0;
10'b110001: pixel = 10'b0;
10'b110010: pixel = 10'b0;
10'b110011: pixel = 10'b0;
10'b110100: pixel = 10'b0;
10'b110101: pixel = 10'b0;
10'b110110: pixel = 10'b0;
10'b110111: pixel = 10'b0;
10'b111000: pixel = 10'b0;
10'b111001: pixel = 10'b0;
10'b111010: pixel = 10'b0;
10'b111011: pixel = 10'b0;
10'b111100: pixel = 10'b0;
10'b111101: pixel = 10'b0;
10'b111110: pixel = 10'b0;
10'b111111: pixel = 10'b0;
10'b1000000: pixel = 10'b0;
10'b1000001: pixel = 10'b0;
10'b1000010: pixel = 10'b0;
10'b1000011: pixel = 10'b0;
10'b1000100: pixel = 10'b0;
10'b1000101: pixel = 10'b0;
10'b1000110: pixel = 10'b0;
10'b1000111: pixel = 10'b0;
10'b1001000: pixel = 10'b0;
10'b1001001: pixel = 10'b0;
10'b1001010: pixel = 10'b0;
10'b1001011: pixel = 10'b0;
10'b1001100: pixel = 10'b0;
10'b1001101: pixel = 10'b0;
10'b1001110: pixel = 10'b0;
10'b1001111: pixel = 10'b0;
10'b1010000: pixel = 10'b0;
10'b1010001: pixel = 10'b0;
10'b1010010: pixel = 10'b0;
10'b1010011: pixel = 10'b0;
10'b1010100: pixel = 10'b0;
10'b1010101: pixel = 10'b0;
10'b1010110: pixel = 10'b0;
10'b1010111: pixel = 10'b0;
10'b1011000: pixel = 10'b0;
10'b1011001: pixel = 10'b0;
10'b1011010: pixel = 10'b0;
10'b1011011: pixel = 10'b1;
10'b1011100: pixel = 10'b0;
10'b1011101: pixel = 10'b0;
10'b1011110: pixel = 10'b0;
10'b1011111: pixel = 10'b0;
10'b1100000: pixel = 10'b0;
10'b1100001: pixel = 10'b0;
10'b1100010: pixel = 10'b0;
10'b1100011: pixel = 10'b0;
10'b1100100: pixel = 10'b0;
10'b1100101: pixel = 10'b0;
10'b1100110: pixel = 10'b0;
10'b1100111: pixel = 10'b0;
10'b1101000: pixel = 10'b0;
10'b1101001: pixel = 10'b0;
10'b1101010: pixel = 10'b0;
10'b1101011: pixel = 10'b0;
10'b1101100: pixel = 10'b0;
10'b1101101: pixel = 10'b0;
10'b1101110: pixel = 10'b0;
10'b1101111: pixel = 10'b0;
10'b1110000: pixel = 10'b0;
10'b1110001: pixel = 10'b0;
10'b1110010: pixel = 10'b0;
10'b1110011: pixel = 10'b0;
10'b1110100: pixel = 10'b0;
10'b1110101: pixel = 10'b1100;
10'b1110110: pixel = 10'b10001001;
10'b1110111: pixel = 10'b11101110;
10'b1111000: pixel = 10'b10011000;
10'b1111001: pixel = 10'b1001;
10'b1111010: pixel = 10'b0;
10'b1111011: pixel = 10'b0;
10'b1111100: pixel = 10'b0;
10'b1111101: pixel = 10'b0;
10'b1111110: pixel = 10'b0;
10'b1111111: pixel = 10'b0;
10'b10000000: pixel = 10'b0;
10'b10000001: pixel = 10'b0;
10'b10000010: pixel = 10'b0;
10'b10000011: pixel = 10'b0;
10'b10000100: pixel = 10'b0;
10'b10000101: pixel = 10'b0;
10'b10000110: pixel = 10'b0;
10'b10000111: pixel = 10'b0;
10'b10001000: pixel = 10'b0;
10'b10001001: pixel = 10'b0;
10'b10001010: pixel = 10'b0;
10'b10001011: pixel = 10'b0;
10'b10001100: pixel = 10'b0;
10'b10001101: pixel = 10'b0;
10'b10001110: pixel = 10'b0;
10'b10001111: pixel = 10'b0;
10'b10010000: pixel = 10'b0;
10'b10010001: pixel = 10'b1111010;
10'b10010010: pixel = 10'b11111111;
10'b10010011: pixel = 10'b11111111;
10'b10010100: pixel = 10'b11111111;
10'b10010101: pixel = 10'b1010111;
10'b10010110: pixel = 10'b0;
10'b10010111: pixel = 10'b0;
10'b10011000: pixel = 10'b0;
10'b10011001: pixel = 10'b0;
10'b10011010: pixel = 10'b0;
10'b10011011: pixel = 10'b0;
10'b10011100: pixel = 10'b0;
10'b10011101: pixel = 10'b0;
10'b10011110: pixel = 10'b0;
10'b10011111: pixel = 10'b0;
10'b10100000: pixel = 10'b0;
10'b10100001: pixel = 10'b0;
10'b10100010: pixel = 10'b0;
10'b10100011: pixel = 10'b0;
10'b10100100: pixel = 10'b0;
10'b10100101: pixel = 10'b0;
10'b10100110: pixel = 10'b0;
10'b10100111: pixel = 10'b0;
10'b10101000: pixel = 10'b0;
10'b10101001: pixel = 10'b0;
10'b10101010: pixel = 10'b0;
10'b10101011: pixel = 10'b0;
10'b10101100: pixel = 10'b0;
10'b10101101: pixel = 10'b1000110;
10'b10101110: pixel = 10'b11111111;
10'b10101111: pixel = 10'b11111111;
10'b10110000: pixel = 10'b11111111;
10'b10110001: pixel = 10'b1010000;
10'b10110010: pixel = 10'b0;
10'b10110011: pixel = 10'b0;
10'b10110100: pixel = 10'b0;
10'b10110101: pixel = 10'b0;
10'b10110110: pixel = 10'b0;
10'b10110111: pixel = 10'b0;
10'b10111000: pixel = 10'b0;
10'b10111001: pixel = 10'b0;
10'b10111010: pixel = 10'b0;
10'b10111011: pixel = 10'b0;
10'b10111100: pixel = 10'b0;
10'b10111101: pixel = 10'b10;
10'b10111110: pixel = 10'b100010;
10'b10111111: pixel = 10'b0;
10'b11000000: pixel = 10'b0;
10'b11000001: pixel = 10'b0;
10'b11000010: pixel = 10'b0;
10'b11000011: pixel = 10'b0;
10'b11000100: pixel = 10'b0;
10'b11000101: pixel = 10'b0;
10'b11000110: pixel = 10'b0;
10'b11000111: pixel = 10'b0;
10'b11001000: pixel = 10'b0;
10'b11001001: pixel = 10'b1100;
10'b11001010: pixel = 10'b11111000;
10'b11001011: pixel = 10'b11111111;
10'b11001100: pixel = 10'b11111111;
10'b11001101: pixel = 10'b1010010;
10'b11001110: pixel = 10'b0;
10'b11001111: pixel = 10'b0;
10'b11010000: pixel = 10'b0;
10'b11010001: pixel = 10'b0;
10'b11010010: pixel = 10'b0;
10'b11010011: pixel = 10'b0;
10'b11010100: pixel = 10'b0;
10'b11010101: pixel = 10'b0;
10'b11010110: pixel = 10'b0;
10'b11010111: pixel = 10'b0;
10'b11011000: pixel = 10'b0;
10'b11011001: pixel = 10'b1000;
10'b11011010: pixel = 10'b11010110;
10'b11011011: pixel = 10'b1101110;
10'b11011100: pixel = 10'b0;
10'b11011101: pixel = 10'b0;
10'b11011110: pixel = 10'b0;
10'b11011111: pixel = 10'b0;
10'b11100000: pixel = 10'b0;
10'b11100001: pixel = 10'b0;
10'b11100010: pixel = 10'b0;
10'b11100011: pixel = 10'b0;
10'b11100100: pixel = 10'b0;
10'b11100101: pixel = 10'b0;
10'b11100110: pixel = 10'b11010010;
10'b11100111: pixel = 10'b11111111;
10'b11101000: pixel = 10'b11111111;
10'b11101001: pixel = 10'b10111001;
10'b11101010: pixel = 10'b1111011;
10'b11101011: pixel = 10'b10110;
10'b11101100: pixel = 10'b0;
10'b11101101: pixel = 10'b0;
10'b11101110: pixel = 10'b0;
10'b11101111: pixel = 10'b0;
10'b11110000: pixel = 10'b0;
10'b11110001: pixel = 10'b0;
10'b11110010: pixel = 10'b0;
10'b11110011: pixel = 10'b0;
10'b11110100: pixel = 10'b0;
10'b11110101: pixel = 10'b0;
10'b11110110: pixel = 10'b1010011;
10'b11110111: pixel = 10'b11110100;
10'b11111000: pixel = 10'b1101000;
10'b11111001: pixel = 10'b0;
10'b11111010: pixel = 10'b0;
10'b11111011: pixel = 10'b0;
10'b11111100: pixel = 10'b0;
10'b11111101: pixel = 10'b0;
10'b11111110: pixel = 10'b0;
10'b11111111: pixel = 10'b0;
10'b100000000: pixel = 10'b0;
10'b100000001: pixel = 10'b0;
10'b100000010: pixel = 10'b10011001;
10'b100000011: pixel = 10'b11111111;
10'b100000100: pixel = 10'b11111111;
10'b100000101: pixel = 10'b11111111;
10'b100000110: pixel = 10'b11111111;
10'b100000111: pixel = 10'b11110011;
10'b100001000: pixel = 10'b101111;
10'b100001001: pixel = 10'b0;
10'b100001010: pixel = 10'b0;
10'b100001011: pixel = 10'b0;
10'b100001100: pixel = 10'b0;
10'b100001101: pixel = 10'b0;
10'b100001110: pixel = 10'b0;
10'b100001111: pixel = 10'b0;
10'b100010000: pixel = 10'b0;
10'b100010001: pixel = 10'b0;
10'b100010010: pixel = 10'b110;
10'b100010011: pixel = 10'b11011111;
10'b100010100: pixel = 10'b11110100;
10'b100010101: pixel = 10'b111100;
10'b100010110: pixel = 10'b0;
10'b100010111: pixel = 10'b0;
10'b100011000: pixel = 10'b100100;
10'b100011001: pixel = 10'b1101010;
10'b100011010: pixel = 10'b101000;
10'b100011011: pixel = 10'b11100;
10'b100011100: pixel = 10'b10010;
10'b100011101: pixel = 10'b100111;
10'b100011110: pixel = 10'b10010100;
10'b100011111: pixel = 10'b11111111;
10'b100100000: pixel = 10'b11111111;
10'b100100001: pixel = 10'b11111111;
10'b100100010: pixel = 10'b11111101;
10'b100100011: pixel = 10'b10100111;
10'b100100100: pixel = 10'b1010;
10'b100100101: pixel = 10'b0;
10'b100100110: pixel = 10'b0;
10'b100100111: pixel = 10'b0;
10'b100101000: pixel = 10'b0;
10'b100101001: pixel = 10'b0;
10'b100101010: pixel = 10'b0;
10'b100101011: pixel = 10'b0;
10'b100101100: pixel = 10'b0;
10'b100101101: pixel = 10'b0;
10'b100101110: pixel = 10'b0;
10'b100101111: pixel = 10'b1101011;
10'b100110000: pixel = 10'b11111110;
10'b100110001: pixel = 10'b11110001;
10'b100110010: pixel = 10'b110010;
10'b100110011: pixel = 10'b0;
10'b100110100: pixel = 10'b10110;
10'b100110101: pixel = 10'b10100011;
10'b100110110: pixel = 10'b11011010;
10'b100110111: pixel = 10'b11111110;
10'b100111000: pixel = 10'b11111000;
10'b100111001: pixel = 10'b11111110;
10'b100111010: pixel = 10'b11111111;
10'b100111011: pixel = 10'b11111111;
10'b100111100: pixel = 10'b11110111;
10'b100111101: pixel = 10'b10100100;
10'b100111110: pixel = 10'b1001101;
10'b100111111: pixel = 10'b1000;
10'b101000000: pixel = 10'b0;
10'b101000001: pixel = 10'b0;
10'b101000010: pixel = 10'b0;
10'b101000011: pixel = 10'b0;
10'b101000100: pixel = 10'b0;
10'b101000101: pixel = 10'b0;
10'b101000110: pixel = 10'b0;
10'b101000111: pixel = 10'b0;
10'b101001000: pixel = 10'b0;
10'b101001001: pixel = 10'b0;
10'b101001010: pixel = 10'b0;
10'b101001011: pixel = 10'b110;
10'b101001100: pixel = 10'b11100000;
10'b101001101: pixel = 10'b11111110;
10'b101001110: pixel = 10'b11001001;
10'b101001111: pixel = 10'b1111;
10'b101010000: pixel = 10'b0;
10'b101010001: pixel = 10'b10000;
10'b101010010: pixel = 10'b11010010;
10'b101010011: pixel = 10'b11101000;
10'b101010100: pixel = 10'b11111111;
10'b101010101: pixel = 10'b11111111;
10'b101010110: pixel = 10'b11111111;
10'b101010111: pixel = 10'b11111111;
10'b101011000: pixel = 10'b10111000;
10'b101011001: pixel = 10'b11;
10'b101011010: pixel = 10'b0;
10'b101011011: pixel = 10'b0;
10'b101011100: pixel = 10'b0;
10'b101011101: pixel = 10'b0;
10'b101011110: pixel = 10'b0;
10'b101011111: pixel = 10'b0;
10'b101100000: pixel = 10'b0;
10'b101100001: pixel = 10'b0;
10'b101100010: pixel = 10'b0;
10'b101100011: pixel = 10'b0;
10'b101100100: pixel = 10'b0;
10'b101100101: pixel = 10'b0;
10'b101100110: pixel = 10'b0;
10'b101100111: pixel = 10'b0;
10'b101101000: pixel = 10'b1110110;
10'b101101001: pixel = 10'b11111000;
10'b101101010: pixel = 10'b11111100;
10'b101101011: pixel = 10'b10000101;
10'b101101100: pixel = 10'b0;
10'b101101101: pixel = 10'b0;
10'b101101110: pixel = 10'b1010;
10'b101101111: pixel = 10'b10001010;
10'b101110000: pixel = 10'b11001110;
10'b101110001: pixel = 10'b10000100;
10'b101110010: pixel = 10'b10010010;
10'b101110011: pixel = 10'b11111111;
10'b101110100: pixel = 10'b10100111;
10'b101110101: pixel = 10'b0;
10'b101110110: pixel = 10'b0;
10'b101110111: pixel = 10'b0;
10'b101111000: pixel = 10'b0;
10'b101111001: pixel = 10'b0;
10'b101111010: pixel = 10'b0;
10'b101111011: pixel = 10'b0;
10'b101111100: pixel = 10'b0;
10'b101111101: pixel = 10'b0;
10'b101111110: pixel = 10'b0;
10'b101111111: pixel = 10'b0;
10'b110000000: pixel = 10'b0;
10'b110000001: pixel = 10'b0;
10'b110000010: pixel = 10'b0;
10'b110000011: pixel = 10'b0;
10'b110000100: pixel = 10'b10100;
10'b110000101: pixel = 10'b11101000;
10'b110000110: pixel = 10'b11110100;
10'b110000111: pixel = 10'b11011000;
10'b110001000: pixel = 10'b0;
10'b110001001: pixel = 10'b0;
10'b110001010: pixel = 10'b0;
10'b110001011: pixel = 10'b1001;
10'b110001100: pixel = 10'b111;
10'b110001101: pixel = 10'b0;
10'b110001110: pixel = 10'b111100;
10'b110001111: pixel = 10'b11111111;
10'b110010000: pixel = 10'b10101111;
10'b110010001: pixel = 10'b0;
10'b110010010: pixel = 10'b0;
10'b110010011: pixel = 10'b0;
10'b110010100: pixel = 10'b0;
10'b110010101: pixel = 10'b0;
10'b110010110: pixel = 10'b0;
10'b110010111: pixel = 10'b0;
10'b110011000: pixel = 10'b0;
10'b110011001: pixel = 10'b0;
10'b110011010: pixel = 10'b0;
10'b110011011: pixel = 10'b0;
10'b110011100: pixel = 10'b0;
10'b110011101: pixel = 10'b0;
10'b110011110: pixel = 10'b0;
10'b110011111: pixel = 10'b0;
10'b110100000: pixel = 10'b0;
10'b110100001: pixel = 10'b10001101;
10'b110100010: pixel = 10'b11011010;
10'b110100011: pixel = 10'b10100010;
10'b110100100: pixel = 10'b0;
10'b110100101: pixel = 10'b0;
10'b110100110: pixel = 10'b0;
10'b110100111: pixel = 10'b0;
10'b110101000: pixel = 10'b0;
10'b110101001: pixel = 10'b0;
10'b110101010: pixel = 10'b1101010;
10'b110101011: pixel = 10'b11111111;
10'b110101100: pixel = 10'b11101000;
10'b110101101: pixel = 10'b1100101;
10'b110101110: pixel = 10'b10000101;
10'b110101111: pixel = 10'b1010000;
10'b110110000: pixel = 10'b100110;
10'b110110001: pixel = 10'b0;
10'b110110010: pixel = 10'b0;
10'b110110011: pixel = 10'b0;
10'b110110100: pixel = 10'b0;
10'b110110101: pixel = 10'b0;
10'b110110110: pixel = 10'b0;
10'b110110111: pixel = 10'b0;
10'b110111000: pixel = 10'b0;
10'b110111001: pixel = 10'b0;
10'b110111010: pixel = 10'b0;
10'b110111011: pixel = 10'b0;
10'b110111100: pixel = 10'b1111;
10'b110111101: pixel = 10'b10111100;
10'b110111110: pixel = 10'b11111100;
10'b110111111: pixel = 10'b10111100;
10'b111000000: pixel = 10'b0;
10'b111000001: pixel = 10'b0;
10'b111000010: pixel = 10'b0;
10'b111000011: pixel = 10'b0;
10'b111000100: pixel = 10'b0;
10'b111000101: pixel = 10'b100101;
10'b111000110: pixel = 10'b11001001;
10'b111000111: pixel = 10'b11111111;
10'b111001000: pixel = 10'b11111111;
10'b111001001: pixel = 10'b11111111;
10'b111001010: pixel = 10'b11111111;
10'b111001011: pixel = 10'b11111111;
10'b111001100: pixel = 10'b11111011;
10'b111001101: pixel = 10'b1100000;
10'b111001110: pixel = 10'b0;
10'b111001111: pixel = 10'b0;
10'b111010000: pixel = 10'b0;
10'b111010001: pixel = 10'b0;
10'b111010010: pixel = 10'b0;
10'b111010011: pixel = 10'b0;
10'b111010100: pixel = 10'b0;
10'b111010101: pixel = 10'b0;
10'b111010110: pixel = 10'b0;
10'b111010111: pixel = 10'b10000;
10'b111011000: pixel = 10'b11000010;
10'b111011001: pixel = 10'b11111111;
10'b111011010: pixel = 10'b11110100;
10'b111011011: pixel = 10'b1000010;
10'b111011100: pixel = 10'b0;
10'b111011101: pixel = 10'b10100;
10'b111011110: pixel = 10'b1101110;
10'b111011111: pixel = 10'b10001111;
10'b111100000: pixel = 10'b11011101;
10'b111100001: pixel = 10'b11111111;
10'b111100010: pixel = 10'b11111111;
10'b111100011: pixel = 10'b11111111;
10'b111100100: pixel = 10'b11111111;
10'b111100101: pixel = 10'b11100010;
10'b111100110: pixel = 10'b10111101;
10'b111100111: pixel = 10'b11111111;
10'b111101000: pixel = 10'b11111110;
10'b111101001: pixel = 10'b11110010;
10'b111101010: pixel = 10'b1011;
10'b111101011: pixel = 10'b0;
10'b111101100: pixel = 10'b0;
10'b111101101: pixel = 10'b0;
10'b111101110: pixel = 10'b0;
10'b111101111: pixel = 10'b0;
10'b111110000: pixel = 10'b0;
10'b111110001: pixel = 10'b0;
10'b111110010: pixel = 10'b101011;
10'b111110011: pixel = 10'b11011001;
10'b111110100: pixel = 10'b11111111;
10'b111110101: pixel = 10'b11110001;
10'b111110110: pixel = 10'b101100;
10'b111110111: pixel = 10'b0;
10'b111111000: pixel = 10'b0;
10'b111111001: pixel = 10'b10110101;
10'b111111010: pixel = 10'b11111100;
10'b111111011: pixel = 10'b11111111;
10'b111111100: pixel = 10'b11111111;
10'b111111101: pixel = 10'b11111111;
10'b111111110: pixel = 10'b11111011;
10'b111111111: pixel = 10'b10100100;
10'b1000000000: pixel = 10'b1101000;
10'b1000000001: pixel = 10'b111;
10'b1000000010: pixel = 10'b11;
10'b1000000011: pixel = 10'b11011010;
10'b1000000100: pixel = 10'b11111111;
10'b1000000101: pixel = 10'b11111111;
10'b1000000110: pixel = 10'b110101;
10'b1000000111: pixel = 10'b0;
10'b1000001000: pixel = 10'b0;
10'b1000001001: pixel = 10'b0;
10'b1000001010: pixel = 10'b0;
10'b1000001011: pixel = 10'b0;
10'b1000001100: pixel = 10'b0;
10'b1000001101: pixel = 10'b1010011;
10'b1000001110: pixel = 10'b11101100;
10'b1000001111: pixel = 10'b11111111;
10'b1000010000: pixel = 10'b11101111;
10'b1000010001: pixel = 10'b1101000;
10'b1000010010: pixel = 10'b1;
10'b1000010011: pixel = 10'b0;
10'b1000010100: pixel = 10'b1001110;
10'b1000010101: pixel = 10'b11100001;
10'b1000010110: pixel = 10'b11111111;
10'b1000010111: pixel = 10'b11111111;
10'b1000011000: pixel = 10'b11111111;
10'b1000011001: pixel = 10'b11011111;
10'b1000011010: pixel = 10'b101010;
10'b1000011011: pixel = 10'b0;
10'b1000011100: pixel = 10'b0;
10'b1000011101: pixel = 10'b0;
10'b1000011110: pixel = 10'b11;
10'b1000011111: pixel = 10'b11010111;
10'b1000100000: pixel = 10'b11111111;
10'b1000100001: pixel = 10'b11100100;
10'b1000100010: pixel = 10'b11011;
10'b1000100011: pixel = 10'b0;
10'b1000100100: pixel = 10'b0;
10'b1000100101: pixel = 10'b0;
10'b1000100110: pixel = 10'b0;
10'b1000100111: pixel = 10'b0;
10'b1000101000: pixel = 10'b1010001;
10'b1000101001: pixel = 10'b11111001;
10'b1000101010: pixel = 10'b11110000;
10'b1000101011: pixel = 10'b10100100;
10'b1000101100: pixel = 10'b101001;
10'b1000101101: pixel = 10'b0;
10'b1000101110: pixel = 10'b0;
10'b1000101111: pixel = 10'b0;
10'b1000110000: pixel = 10'b11000011;
10'b1000110001: pixel = 10'b11111111;
10'b1000110010: pixel = 10'b11101011;
10'b1000110011: pixel = 10'b11111110;
10'b1000110100: pixel = 10'b1011100;
10'b1000110101: pixel = 10'b11001;
10'b1000110110: pixel = 10'b11100;
10'b1000110111: pixel = 10'b1000001;
10'b1000111000: pixel = 10'b1101000;
10'b1000111001: pixel = 10'b10001110;
10'b1000111010: pixel = 10'b11101001;
10'b1000111011: pixel = 10'b11111111;
10'b1000111100: pixel = 10'b11111001;
10'b1000111101: pixel = 10'b1110000;
10'b1000111110: pixel = 10'b0;
10'b1000111111: pixel = 10'b0;
10'b1001000000: pixel = 10'b0;
10'b1001000001: pixel = 10'b0;
10'b1001000010: pixel = 10'b1110;
10'b1001000011: pixel = 10'b10011011;
10'b1001000100: pixel = 10'b11111000;
10'b1001000101: pixel = 10'b11101011;
10'b1001000110: pixel = 10'b1111010;
10'b1001000111: pixel = 10'b1001;
10'b1001001000: pixel = 10'b0;
10'b1001001001: pixel = 10'b0;
10'b1001001010: pixel = 10'b0;
10'b1001001011: pixel = 10'b0;
10'b1001001100: pixel = 10'b101010;
10'b1001001101: pixel = 10'b111011;
10'b1001001110: pixel = 10'b11;
10'b1001001111: pixel = 10'b110001;
10'b1001010000: pixel = 10'b1110010;
10'b1001010001: pixel = 10'b10110011;
10'b1001010010: pixel = 10'b11100000;
10'b1001010011: pixel = 10'b11111110;
10'b1001010100: pixel = 10'b11111100;
10'b1001010101: pixel = 10'b11111000;
10'b1001010110: pixel = 10'b11111111;
10'b1001010111: pixel = 10'b11111101;
10'b1001011000: pixel = 10'b1100011;
10'b1001011001: pixel = 10'b0;
10'b1001011010: pixel = 10'b0;
10'b1001011011: pixel = 10'b0;
10'b1001011100: pixel = 10'b0;
10'b1001011101: pixel = 10'b11111;
10'b1001011110: pixel = 10'b11011111;
10'b1001011111: pixel = 10'b11111111;
10'b1001100000: pixel = 10'b11110111;
10'b1001100001: pixel = 10'b1101100;
10'b1001100010: pixel = 10'b1000;
10'b1001100011: pixel = 10'b0;
10'b1001100100: pixel = 10'b0;
10'b1001100101: pixel = 10'b0;
10'b1001100110: pixel = 10'b0;
10'b1001100111: pixel = 10'b0;
10'b1001101000: pixel = 10'b0;
10'b1001101001: pixel = 10'b0;
10'b1001101010: pixel = 10'b0;
10'b1001101011: pixel = 10'b0;
10'b1001101100: pixel = 10'b0;
10'b1001101101: pixel = 10'b0;
10'b1001101110: pixel = 10'b0;
10'b1001101111: pixel = 10'b110;
10'b1001110000: pixel = 10'b10;
10'b1001110001: pixel = 10'b0;
10'b1001110010: pixel = 10'b11110;
10'b1001110011: pixel = 10'b100011;
10'b1001110100: pixel = 10'b0;
10'b1001110101: pixel = 10'b0;
10'b1001110110: pixel = 10'b0;
10'b1001110111: pixel = 10'b0;
10'b1001111000: pixel = 10'b1110101;
10'b1001111001: pixel = 10'b11101111;
10'b1001111010: pixel = 10'b11110111;
10'b1001111011: pixel = 10'b11011101;
10'b1001111100: pixel = 10'b1111100;
10'b1001111101: pixel = 10'b10;
10'b1001111110: pixel = 10'b0;
10'b1001111111: pixel = 10'b0;
10'b1010000000: pixel = 10'b0;
10'b1010000001: pixel = 10'b0;
10'b1010000010: pixel = 10'b0;
10'b1010000011: pixel = 10'b0;
10'b1010000100: pixel = 10'b0;
10'b1010000101: pixel = 10'b0;
10'b1010000110: pixel = 10'b0;
10'b1010000111: pixel = 10'b0;
10'b1010001000: pixel = 10'b0;
10'b1010001001: pixel = 10'b0;
10'b1010001010: pixel = 10'b0;
10'b1010001011: pixel = 10'b0;
10'b1010001100: pixel = 10'b0;
10'b1010001101: pixel = 10'b0;
10'b1010001110: pixel = 10'b0;
10'b1010001111: pixel = 10'b0;
10'b1010010000: pixel = 10'b0;
10'b1010010001: pixel = 10'b0;
10'b1010010010: pixel = 10'b0;
10'b1010010011: pixel = 10'b10000101;
10'b1010010100: pixel = 10'b11111011;
10'b1010010101: pixel = 10'b11111111;
10'b1010010110: pixel = 10'b11010100;
10'b1010010111: pixel = 10'b110110;
10'b1010011000: pixel = 10'b0;
10'b1010011001: pixel = 10'b0;
10'b1010011010: pixel = 10'b0;
10'b1010011011: pixel = 10'b0;
10'b1010011100: pixel = 10'b0;
10'b1010011101: pixel = 10'b0;
10'b1010011110: pixel = 10'b0;
10'b1010011111: pixel = 10'b0;
10'b1010100000: pixel = 10'b0;
10'b1010100001: pixel = 10'b0;
10'b1010100010: pixel = 10'b0;
10'b1010100011: pixel = 10'b0;
10'b1010100100: pixel = 10'b0;
10'b1010100101: pixel = 10'b0;
10'b1010100110: pixel = 10'b0;
10'b1010100111: pixel = 10'b0;
10'b1010101000: pixel = 10'b0;
10'b1010101001: pixel = 10'b0;
10'b1010101010: pixel = 10'b0;
10'b1010101011: pixel = 10'b0;
10'b1010101100: pixel = 10'b0;
10'b1010101101: pixel = 10'b0;
10'b1010101110: pixel = 10'b1010;
10'b1010101111: pixel = 10'b1000000;
10'b1010110000: pixel = 10'b111111;
10'b1010110001: pixel = 10'b110010;
10'b1010110010: pixel = 10'b10010;
10'b1010110011: pixel = 10'b0;
10'b1010110100: pixel = 10'b0;
10'b1010110101: pixel = 10'b0;
10'b1010110110: pixel = 10'b0;
10'b1010110111: pixel = 10'b0;
10'b1010111000: pixel = 10'b0;
10'b1010111001: pixel = 10'b0;
10'b1010111010: pixel = 10'b0;
10'b1010111011: pixel = 10'b0;
10'b1010111100: pixel = 10'b0;
10'b1010111101: pixel = 10'b0;
10'b1010111110: pixel = 10'b0;
10'b1010111111: pixel = 10'b0;
10'b1011000000: pixel = 10'b0;
10'b1011000001: pixel = 10'b0;
10'b1011000010: pixel = 10'b0;
10'b1011000011: pixel = 10'b0;
10'b1011000100: pixel = 10'b0;
10'b1011000101: pixel = 10'b0;
10'b1011000110: pixel = 10'b0;
10'b1011000111: pixel = 10'b0;
10'b1011001000: pixel = 10'b0;
10'b1011001001: pixel = 10'b0;
10'b1011001010: pixel = 10'b0;
10'b1011001011: pixel = 10'b0;
10'b1011001100: pixel = 10'b0;
10'b1011001101: pixel = 10'b0;
10'b1011001110: pixel = 10'b0;
10'b1011001111: pixel = 10'b0;
10'b1011010000: pixel = 10'b0;
10'b1011010001: pixel = 10'b0;
10'b1011010010: pixel = 10'b0;
10'b1011010011: pixel = 10'b0;
10'b1011010100: pixel = 10'b0;
10'b1011010101: pixel = 10'b0;
10'b1011010110: pixel = 10'b0;
10'b1011010111: pixel = 10'b0;
10'b1011011000: pixel = 10'b0;
10'b1011011001: pixel = 10'b0;
10'b1011011010: pixel = 10'b0;
10'b1011011011: pixel = 10'b0;
10'b1011011100: pixel = 10'b0;
10'b1011011101: pixel = 10'b0;
10'b1011011110: pixel = 10'b0;
10'b1011011111: pixel = 10'b0;
10'b1011100000: pixel = 10'b0;
10'b1011100001: pixel = 10'b0;
10'b1011100010: pixel = 10'b0;
10'b1011100011: pixel = 10'b0;
10'b1011100100: pixel = 10'b0;
10'b1011100101: pixel = 10'b0;
10'b1011100110: pixel = 10'b0;
10'b1011100111: pixel = 10'b0;
10'b1011101000: pixel = 10'b0;
10'b1011101001: pixel = 10'b0;
10'b1011101010: pixel = 10'b0;
10'b1011101011: pixel = 10'b0;
10'b1011101100: pixel = 10'b0;
10'b1011101101: pixel = 10'b0;
10'b1011101110: pixel = 10'b0;
10'b1011101111: pixel = 10'b0;
10'b1011110000: pixel = 10'b0;
10'b1011110001: pixel = 10'b0;
10'b1011110010: pixel = 10'b0;
10'b1011110011: pixel = 10'b0;
10'b1011110100: pixel = 10'b0;
10'b1011110101: pixel = 10'b0;
10'b1011110110: pixel = 10'b0;
10'b1011110111: pixel = 10'b0;
10'b1011111000: pixel = 10'b0;
10'b1011111001: pixel = 10'b0;
10'b1011111010: pixel = 10'b0;
10'b1011111011: pixel = 10'b0;
10'b1011111100: pixel = 10'b0;
10'b1011111101: pixel = 10'b0;
10'b1011111110: pixel = 10'b0;
10'b1011111111: pixel = 10'b0;
10'b1100000000: pixel = 10'b0;
10'b1100000001: pixel = 10'b0;
10'b1100000010: pixel = 10'b0;
10'b1100000011: pixel = 10'b0;
10'b1100000100: pixel = 10'b0;
10'b1100000101: pixel = 10'b0;
10'b1100000110: pixel = 10'b0;
10'b1100000111: pixel = 10'b0;
10'b1100001000: pixel = 10'b0;
10'b1100001001: pixel = 10'b0;
10'b1100001010: pixel = 10'b0;
10'b1100001011: pixel = 10'b0;
10'b1100001100: pixel = 10'b0;
10'b1100001101: pixel = 10'b0;
10'b1100001110: pixel = 10'b0;
10'b1100001111: pixel = 10'b0;
	endcase 
  end 

endmodule : mnist_lut