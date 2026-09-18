# Run from the repository root: Rscript ame/plot_ame.R
library(ggplot2)
library(dplyr)
library(ggtext)
library(patchwork)
if (.Platform$OS.type == "windows") invisible(Sys.setlocale("LC_CTYPE", "English_United States.utf8"))

cdtally_dir <- "outputs/cdtally/ame"
lamp_dir <- "outputs/lamp_return/ame"
output_dir <- "outputs/figures_tables/ame"

write_table <- function(x, path) write.csv(x, path, row.names = FALSE, na = "NA")
edge_key <- function(source, target) paste(source, target, sep = "\r")

receiver_panel <- function(bundle) {
  nodes <- bundle$results$nodes %>%
    filter(observed_in_dyads > 0) %>% arrange(desc(abs(receiver_effect)))
  selected <- unique(c(head(nodes$node, 15), nodes$node[nodes$is_key_action]))
  dat <- nodes %>% filter(node %in% selected) %>% arrange(receiver_effect) %>%
    mutate(node_factor = factor(node, levels = node),
      key_label = factor(ifelse(is_key_action, "Key action", "Non-key action"),
        levels = c("Key action", "Non-key action")))
  labels <- setNames(ifelse(dat$is_key_action, paste0("**", dat$node, "**"), dat$node), dat$node)
  ggplot(dat, aes(node_factor, receiver_effect, fill = key_label)) +
    geom_col() + geom_hline(yintercept = 0, linetype = "dashed", linewidth = .4) +
    coord_flip() + scale_x_discrete(labels = labels) + scale_fill_discrete(name = NULL) +
    labs(x = NULL, y = expression("Receiver effect, " * b[j])) + theme_bw() +
    theme(axis.text.y = element_markdown(), legend.position = "inside",
      legend.position.inside = c(.98, .03), legend.justification = c(1, 0),
      legend.background = element_rect(fill = "white", colour = "grey70", linewidth = .3),
      legend.margin = margin(4, 5, 4, 5))
}

pathway_panel <- function(bundle) {
  dat <- bundle$results$pathways %>% filter(key_to_key) %>% arrange(mult_inner) %>%
    mutate(edge = paste0("**", source, "** \u2192 **", target, "**"),
      edge = factor(edge, levels = edge))
  ggplot(dat, aes(edge, mult_inner)) + geom_col() +
    geom_hline(yintercept = 0, linetype = "dashed", linewidth = .4) + coord_flip() +
    labs(x = NULL, y = expression("Multiplicative effect, " * u[i]^T * v[j])) +
    theme_bw() + theme(axis.text.y = element_markdown())
}

heatmap_panel <- function(bundle) {
  keys <- bundle$keys
  p <- bundle$results$pathways
  dat <- expand.grid(source = keys, target = keys, stringsAsFactors = FALSE)
  ix <- match(edge_key(dat$source, dat$target), edge_key(p$source, p$target))
  dat$value <- p$mult_inner[ix]
  dat$source <- factor(dat$source, levels = rev(keys))
  dat$target <- factor(dat$target, levels = keys)
  labels <- setNames(sub("^wb_pg_8_4_", "", keys), keys)
  lim <- max(abs(dat$value), na.rm = TRUE)
  ggplot(dat, aes(target, source, fill = value)) + geom_tile(colour = "white", linewidth = .2) +
    scale_fill_gradient2(low = "#2166AC", mid = "white", high = "#B2182B", midpoint = 0,
      limits = c(-lim, lim), na.value = "#D9D9D9", name = "Multiplicative\neffect") +
    scale_x_discrete(labels = labels, drop = FALSE) + scale_y_discrete(labels = labels, drop = FALSE) +
    coord_fixed() + labs(title = "Multiplicative effects among key actions", x = "Target action", y = "Source action") +
    theme_minimal(base_size = 10) + theme(panel.grid = element_blank(),
      plot.title = element_text(size = 12, face = "bold"), axis.text.x = element_text(angle = 45, hjust = 1),
      legend.position = "right", plot.margin = margin(8, 5, 5, 5))
}

save_plot <- function(plot, filename, width, height) {
  ggsave(file.path(output_dir, paste0(filename, ".png")), plot,
    width = width, height = height, dpi = 300, bg = "white")
  ggsave(file.path(output_dir, paste0(filename, ".pdf")), plot,
    width = width, height = height, device = cairo_pdf, bg = "white")
}

specs <- data.frame(id = c("cdtally_AT", "cdtally_SK_GB", "lamp_GB", "lamp_SK_GB"),
  directory = c(cdtally_dir, cdtally_dir, lamp_dir, lamp_dir),
  kind = c("bar", "bar", "heatmap", "heatmap"), width = c(7, 7, 8.4, 8.4),
  receiver_height = c(5, 5, 4.65, 4.65), mult_height = c(5, 5, 6.15, 6.15),
  mult_suffix = c("mult", "mult", "heatmap", "heatmap"))
mult_function <- list(bar = pathway_panel, heatmap = heatmap_panel)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
panels <- selected <- list()

for (i in seq_len(nrow(specs))) {
  spec <- specs[i, ]
  path <- file.path(spec$directory, spec$id)
  bundle <- readRDS(file.path(path, "effects.rds"))
  receiver <- receiver_panel(bundle)
  if (spec$kind == "heatmap") {
    receiver <- receiver + labs(title = "Receiver effects") +
      theme(plot.title = element_text(size = 12, face = "bold"),
        axis.text.y = element_markdown(size = 9), axis.text.x = element_text(size = 9))
  }
  mult <- mult_function[[spec$kind]](bundle)
  save_plot(receiver, paste0(spec$id, "_receiver_effect"), spec$width, spec$receiver_height)
  save_plot(mult, paste0(spec$id, "_key_to_key_", spec$mult_suffix), spec$width, spec$mult_height)
  write_table(receiver$data, file.path(output_dir, paste0(spec$id, "_receiver_data.csv")))
  write_table(mult$data, file.path(output_dir, paste0(spec$id, "_multiplicative_data.csv")))
  panels[[spec$id]] <- list(receiver = receiver, multiplicative = mult)
  p <- bundle$results$pathways
  selected[[spec$id]] <- cbind(comparison = spec$id, p[p$paper_example, ])

}

combine_panels <- function(ids) {
  (panels[[ids[1]]]$receiver | panels[[ids[1]]]$multiplicative) /
    (panels[[ids[2]]]$receiver | panels[[ids[2]]]$multiplicative) +
    plot_annotation(tag_levels = "a", tag_prefix = "(", tag_suffix = ")")
}
save_plot(combine_panels(specs$id[1:2]), "CDTally_AME_main", 14, 10)
save_plot(combine_panels(specs$id[3:4]), "LampReturn_AME_supplement", 17, 13)
write_table(bind_rows(selected), file.path(output_dir, "AME_main_text_pathways.csv"))
message("Figures and tables saved to ", output_dir)
