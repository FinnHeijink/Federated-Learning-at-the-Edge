module top_conv (
  input logic           clk_i,
  input logic           rst_ni,
  input logic [107:0]     k_val,
  input logic [7:0] pixel_i,
  input logic           pix_data_valid,
  output logic [7:0]    pixel_o,
  output logic          conv_finished
);

  filter_fsm #(
  .width(12), 
  .input_width(8), 
  .im_dim(28),
  .k_size(9),// 3x3
  .k_dim(3) // 3x3 kernel
  ) machine (
    .k_val(k_val),
    .clk_i(clk_i),
    .rst_ni(rst_ni),  
    .pixel_i(pixel_i),
    .pix_data_valid(pix_data_valid),
    .conv_finished(conv_finished),
    .pixel_o(pixel_o)
  );

endmodule : top_conv