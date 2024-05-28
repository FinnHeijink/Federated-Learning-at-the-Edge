module kernel_weight #(
  parameter width = 16,
  parameter k_size = 9, // 3x3
  parameter kernel_weight_file = "kernel.mif"
)(
  input logic   clk_i,
  input logic   rst_ni,
  // input logic   w_en, // Extraneous write-enable signals
  input logic   r_en, // Read-enable
  input logic  [$clog2(k_size)-1:0] address, // 9-bit vector, $clog2 = ceil 2-log
  output logic [width-1:0] k_out
);

  logic [width-1:0] kernel [$clog2(k_size)-1:0]; // Can we read out entire kernel in 1 clock cycle? 
  
  initial begin
    $readmemb(kernel_weight_file, kernel);
  end
  
  always_ff @(posedge clk_i) begin  // BRAM inferred by clocking read-out
    if (r_en) begin
      k_out <= kernel[address];
    end else begin
      k_out <= '0; 
    end
  end


endmodule : kernel_weight