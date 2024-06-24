module grayscale_conv #(
  parameter width = 12,
  parameter input_width = 8  
  )(
  input logic                  clk_i,
  input logic                  reset,

  input logic [2:0][2:0][input_width-1:0]  pixel_i, // 3x3 input pixel-grid
  input logic [2:0][2:0][width-1:0] k_i,     // 3x3 kernel
  
  input logic                  data_valid,

  output logic [7:0]           pixel_o,  // output pixel
  output logic                 output_valid
);

  // -----------------------
  // Declare local parameters for ease of use/reconfigurability 
  // -----------------------

  // localparam KERNEL_WIDTH = 12;
  localparam PIXEL_WIDTH = 8;
  localparam MULTIPLICATION_WIDTH = 20;
  localparam ACCUMULATOR_WIDTH = 24;
  localparam DECIMAL_WIDTH = 4;

  // -----------------------
  // Multiplier
  // -----------------------

  logic [2:0][2:0][MULTIPLICATION_WIDTH-1:0] mul_q, mul_d;

  const logic [MULTIPLICATION_WIDTH-PIXEL_WIDTH-1:0] MUL_PREFIX = '0; // Prepend signed representation of pixel with sufficient 0s

  always_comb begin
    for (int x = 0; x < 3; x++) begin
      for (int y = 0; y < 3; y++) begin
        mul_d[x][y] = $signed(k_i[x][y]) * $signed({MUL_PREFIX, pixel_i[x][y]});
      end
    end

  end

  logic mul_data_valid;

  always_ff @(posedge clk_i or posedge reset) begin
    if (reset) begin
      mul_q <= '0;
      mul_data_valid <= 1'b0;
    end else begin
      mul_q <= mul_d;
      mul_data_valid <= data_valid;   
    end
  end

  // -----------------------
  // Accumulator
  // -----------------------

  logic [ACCUMULATOR_WIDTH-1:0] acc_q, acc_d;

  always_comb begin
    acc_d = '0;
    for (int i = 0; i < 3; i++) begin
      for (int j = 0; j < 3; j++) begin
          acc_d = $signed(acc_d) + $signed(mul_q[i][j]);
      end
    end
  
 /*    --- Full code -- */ /*
    acc_d  = (($signed(mul_q[0][0]) + $signed(mul_q[0][1]))  +
               ($signed(mul_q[0][2])  + $signed(mul_q[1][0]))) +
               (($signed(mul_q[1][1]) + $signed(mul_q[1][2]))  +
               ($signed(mul_q[2][0])  + $signed(mul_q[2][1]))) +
               $signed(mul_q[2][2]);
    */ 
     
  end
  
  
  always_ff @(posedge clk_i or posedge reset) begin
    if (reset) begin
      acc_q <= '0;
      output_valid <= 1'b0;
    end else begin
      acc_q <= acc_d;
      output_valid <= mul_data_valid;
    end
  end

  // -----------------------
  // OUTPUT
  // -----------------------

  // Only output integer part of accumulator as the output pixel
  // Right shift to cut-off decimal part
  logic [ACCUMULATOR_WIDTH-DECIMAL_WIDTH-1:0] int_round_p;

  always_comb begin
    int_round_p = acc_q >> DECIMAL_WIDTH;
  end

  always_comb begin 
    pixel_o = ($signed(int_round_p) < 8'h00) ? 8'h00 : ((int_round_p > 8'hFF) ? 8'hFF : int_round_p); // saturate output, functions as activation function!!
  end

endmodule : grayscale_conv