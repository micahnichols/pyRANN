/*   MLIP is a software for Machine Learning Interatomic Potentials
 *   MLIP is released under the "New BSD License", see the LICENSE file.
 *   Contributors: Alexander Shapeev, Evgeny Podryabinkin, Ivan Novikov
 */

#ifndef MLIP_MLP_COMMANDS_H
#define MLIP_MLP_COMMANDS_H

bool Commands(const std::string& command, std::vector<std::string>& args, std::map<std::string, std::string>& opts);

#endif // MLIP_MLP_COMMANDS_H