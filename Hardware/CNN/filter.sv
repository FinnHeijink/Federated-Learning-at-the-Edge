module filter (
  input logic                  clk_i,
  input logic                  rst_ni,

  // input logic [1:0]            switch_i,
  
  input  logic [2:0][2:0][7:0] pixel_i,
  
  input logic [2:0][2:0][15:0] k,  
  
  input logic                  data_valid,

  output logic [7:0]           pixel_o,
  output logic                 output_valid
);

  /*
  // Convolution weights
  
  localparam KERNEL_IDENTITY = 2'b00;
  localparam KERNEL_GAUSSIAN = 2'b01;
  localparam KERNEL_SHARP    = 2'b10;
  localparam KERNEL_SOBEL    = 2'b11;
  */
  
  /*
  always_comb begin
    // Default value
    k = '0;

    case (switch_i)
      KERNEL_IDENTITY: begin
        // Identity kernel
        k[0] = {12'd0,  12'd0, 12'd0};
        k[1] = {12'd0, 12'd16, 12'd0};
        k[2] = {12'd0,  12'd0, 12'd0};
      end

      KERNEL_GAUSSIAN: begin
        // Gaussian kernel
        k[0] = {12'd1, 12'd2, 12'd1};
        k[1] = {12'd2, 12'd4, 12'd2};
        k[2] = {12'd1, 12'd2, 12'd1};
      end

      KERNEL_SHARP: begin
        // Sharpening kernel
        k[0] = {  12'd0, -12'd16,   12'd0};
        k[1] = {-12'd16,  12'd80, -12'd16};
        k[2] = {  12'd0, -12'd16,   12'd0};
      end

      KERNEL_SOBEL: begin
        // Sobel kernel
        k[0] = {12'd16, 12'd0, -12'd16};
        k[1] = {12'd32, 12'd0, -12'd32};
        k[2] = {12'd16, 12'd0, -12'd16};
      end
    endcase // case (switch_i[2:1])
  end
  */
  // Convolution block
  logic [7:0] conv_p;

  grayscale_conv pixel_conv (
    .clk_i   (clk_i),
    .rst_ni  (rst_ni),
    
    .data_valid(data_valid),
    .pixel_i (pixel_i),
    .k_i     (k),

    .pixel_o (conv_p),
    .output_valid(output_valid)
    );

  // Output multiplexer
  always_comb begin
    pixel_o = conv_p;
  end


endmodule // rgb_filter
